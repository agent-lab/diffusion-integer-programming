from collections import OrderedDict

import torch
import torch.nn
import torch_sparse
import numpy as np

import math

from functools import partial

from model.utils import default, extract_into_tensor, exists, make_beta_schedule, noise_like, to_torch
from model.utils import make_ddim_timesteps, make_ddim_sampling_parameters
from model.modules import ResidualAttentionBlock



def get_clip_loss(mip_features, x_features):
    device = mip_features.device
    batch_size = mip_features.shape[0]
    mip_features = mip_features.view(batch_size, -1)
    x_features = x_features.view(batch_size, -1)
    logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
    mip_features = mip_features / mip_features.norm(dim=-2, keepdim=True)
    x_features = x_features / x_features.norm(dim=-2, keepdim=True)

    logit_scale = logit_scale.exp()
    logits_per_mip = logit_scale * mip_features @ x_features.t()
    logits_per_x = logits_per_mip.t()
    contrastive_label = torch.arange(batch_size, device=device)

    cross_entropy = torch.nn.CrossEntropyLoss()
    mip_loss = cross_entropy(logits_per_mip, contrastive_label)
    x_loss = cross_entropy(logits_per_x, contrastive_label)
    loss = (mip_loss + x_loss) / 2
    return loss


def get_timestep_embedding(timesteps: torch.Tensor,
                           embedding_dim: int,
                           flip_sin_to_cos: bool = False,
                           downscale_freq_shift: float = 1,
                           scale: float = 1,
                           max_period: int = 10000,
                           ):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb
    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AttnBlock(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.query_linear = torch.nn.Linear(input_dim, input_dim)
        self.key_linear = torch.nn.Linear(input_dim, input_dim)
        self.value_linear = torch.nn.Linear(input_dim, input_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, query, key, value):
        query_transformed = self.query_linear(query)
        key_transformed = self.key_linear(key)
        value_transformed = self.value_linear(value)

        # 计算query和key之间的相似度得分
        scores = torch.matmul(query_transformed, key_transformed.transpose(-2, -1))
        attention_weights = self.relu(scores)

        # 使用attention权重对value进行加权求和
        output = torch.matmul(attention_weights, value_transformed)

        return output


class ResBlock(torch.nn.Module):
    def __init__(self, input_size, emb_size):
        super().__init__()
        self.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, emb_size),
            QuickGELU(),
            torch.nn.Linear(emb_size, emb_size),
            QuickGELU()
        )

    def forward(self, x):
        mlp_x = self.mlp_layer(x)
        out = x + mlp_x
        return out

class NoisePredictV2(torch.nn.Module):
    """
    predict the noise
    input:condition and solution
    output: solution
    """
    def __init__(self, attn_dim: int, n_heads: int, n_layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn_dim = attn_dim
        self.resblocks = torch.nn.ModuleList([ResidualAttentionBlock(attn_dim, n_heads, attn_mask) for _ in range(n_layers)])
        
        # self.mlp_out = torch.nn.Sequential(OrderedDict([
        #             ("linear_1", torch.nn.Linear(attn_dim, attn_dim)),
        #             ("geru", QuickGELU()),
        #             ("linear_2", torch.nn.Linear(attn_dim, attn_dim))
        #             ]))
        # self.ln_noise = torch.nn.LayerNorm(attn_dim)

        # self.mlp_out = torch.nn.Sequential(OrderedDict([
        #             ("linear_1", torch.nn.Linear(attn_dim, attn_dim)),
        #             ("geru", QuickGELU()),
        #             ("linear_2", torch.nn.Linear(attn_dim, attn_dim)),
        #             ("geru", QuickGELU())]))
        # self.ln_mip = torch.nn.LayerNorm(attn_dim)

        self.mlp_xt = torch.nn.Sequential(OrderedDict([
                    ("linear_1", torch.nn.Linear(2*attn_dim, attn_dim)),
                    ("geru", QuickGELU()),
                    ("linear_2", torch.nn.Linear(attn_dim, attn_dim)),
                    ("geru", QuickGELU())
                    ]))


    def forward(self, x_features, t, mip_features, key_padding_mask):
        timestep_embedding = get_timestep_embedding(t, self.attn_dim*2).unsqueeze(dim=-2)
    
        mip_x = torch.concat([mip_features, x_features], dim = -1)

        mip_x_t = torch.concat([timestep_embedding, mip_x], dim = -2)

        mip_x_t = self.mlp_xt(mip_x_t)

        timestep_padding_mask = torch.zeros((mip_features.shape[0],1), dtype=torch.bool, device = mip_x_t.device)
        

        concat_key_padding_mask = torch.concat([timestep_padding_mask, key_padding_mask], dim=-1)

        for module in self.resblocks:
            mip_x_t = module(mip_x_t, concat_key_padding_mask)
        noise = mip_x_t[:, -x_features.shape[1]:, :]

        # noise = noise + self.mlp_out(noise)


        return noise

    def apply_model(self, x_features, t, mip_features, key_padding_mask):
        with torch.no_grad():
            self.eval()
            output = self(x_features, t, mip_features, key_padding_mask)
        return output



class NoisePredict(torch.nn.Module):
    """
    predict the noise
    input:condition and solution
    output: solution
    """
    def __init__(self, var_num, var_dim = 128, mip_dim = 128):
        super().__init__()
        emb_size = 128
        self.var_num = var_num
        start_size = var_dim + mip_dim
        self.attn_layer = AttnBlock(input_dim=var_dim)
        self.condition_emb_layer = torch.nn.Sequential(
            torch.nn.Linear(start_size, emb_size),
            QuickGELU()
        )
        self.resblock1 = ResBlock(input_size=emb_size + 1, emb_size=emb_size + 1)
        self.resblock2 = ResBlock(input_size=emb_size + 1, emb_size=emb_size + 1)
        self.resblock3 = ResBlock(input_size=emb_size + 1, emb_size=emb_size + 1)
        self.mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(emb_size + 1, var_dim),
            QuickGELU(),
            torch.nn.Linear(var_dim, var_dim)
        )

    def forward(self, x, t, condition= None, key_padding_mask = None):
        if condition == None:
            x_con_emb = x
        else:
            # x_con_emb = self.attn_layer(condition, x, x)
            x_con = torch.concat((x, condition), dim=-1)
            x_con_emb = self.condition_emb_layer(x_con)
        timestep_embedding = get_timestep_embedding(t, self.var_num).unsqueeze(dim=-1)
        x_emb = torch.concat((x_con_emb, timestep_embedding), dim=-1)
        x_res = self.resblock1(x_emb)
        x_res = self.resblock2(x_res)
        x_res = self.resblock3(x_res)
        eps = self.mlp_layer(x_res)

        return eps

    def apply_model(self, x, t, condition, key_padding_mask=None):
        with torch.no_grad():
            self.eval()
            output = self(x, t, condition, key_padding_mask=None)
        return output


class DDPMTrainer(torch.nn.Module):
    def __init__(self, attn_dim, n_heads, n_layers, device,
                 timesteps=1000,
                 loss_type="l2",
                 beta_schedule="linear",
                 parameterization="x0"):
        super().__init__()
        self.device = device

        self.loss_type = loss_type
        self.parameterization = parameterization
        self.predict_model = NoisePredictV2(attn_dim, n_heads, n_layers).to(device)
        # self.predict_model = NoisePredict(2000, 128, 128).to(device)
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps)

    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2,
                          cosine_s=8e-3):
        device = self.device
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        # to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('alphas', to_torch(alphas).to(device))
        self.register_buffer('betas', to_torch(betas).to(device))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod).to(device))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev).to(device))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)).to(device))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)).to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)).to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(device))

    def forward(self, x, condition, key_padding_mask):
        # t: tensor[batch_size]
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        x_start, loss = self.p_losses(x, t, condition, key_padding_mask)
        return x_start, loss

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_losses(self, x_start, t, condition, key_padding_mask, noise=None):
        """
        predict the added noise with model, and get the loss
        """
        noise = default(noise, lambda: torch.randn_like(x_start)).to(self.device)
        # x_t= sqrt(alpha)x0 + sqrt(1-alpha)*noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise).to(self.device)
        # predict eps or x0 with unet
        model_out = self.predict_model(x_noisy, t, condition, key_padding_mask)
        if self.parameterization == "eps":
            target = noise
            pred_x_start = self.predict_start_from_noise(x_noisy, t, noise)
        elif self.parameterization == "x0":
            target = x_start
            pred_x_start = model_out
        loss = self.get_loss(model_out.squeeze(), target.squeeze(), key_padding_mask, mean=True)
        return pred_x_start, loss

    def get_loss(self, pred, target, key_padding_mask, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            ## todo: padding target and pred
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred, reduction = 'none')
                loss = loss.mean(dim = -1)
                masked_loss = loss * ~key_padding_mask
                sum_masked_loss = masked_loss.sum(dim = -1)
                count = (~key_padding_mask).sum(dim=-1)
                mean_loss = sum_masked_loss/count
                loss = mean_loss.mean()
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction = 'none')
        # elif self.loss_type == 'bce':
        #     bce_loss_net = torch.nn.BCELoss(reduction='none')
        #     loss = bce_loss_net(pred, target)
        else:
            raise NotImplementedError("unknown loss type '{self.loss_type}'")
        return loss

    def get_predict_model(self):
        return self.predict_model


class DDPMSampler(torch.nn.Module):
    def __init__(self, trainer_model,  decoder = None, gradient_scale = 50000, obj_guided_coef = 0, device = "cpu"):
        super().__init__()
        self.device = device
        self.model = trainer_model

        # self.var_num = self.model.var_num
        # self.var_dim = self.model.var_dim
        # self.mip_dim = self.model.mip_dim
        self.parameterization = self.model.parameterization
        self.predict_model = self.model.get_predict_model()
        self.decoder = decoder
        self.gradient_scale = gradient_scale

        self.obj_guided_coef = obj_guided_coef

        self.v_posterior = 0
        self.original_elbo_weight = 0.
        self.register_schedule()

    def register_schedule(self):
        device = self.device
        betas = self.model.betas
        alphas = self.model.alphas
        alphas_cumprod = self.model.alphas_cumprod
        # hat(alpha_{t-1})
        alphas_cumprod_prev = self.model.alphas_cumprod_prev

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = self.model.linear_start
        self.linear_end = self.model.linear_end

        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'


        self.register_buffer('betas', to_torch(betas).to(device))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod).to(device))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev).to(device))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())).to(device))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())).to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())).to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)).to(device))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance).to(device))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance.cpu(), 1e-20))).to(device))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev.cpu()).to(device) / (1. - alphas_cumprod)).to(device))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas.cpu()).to(device) / (1. - alphas_cumprod)).to(device))


    @torch.no_grad()
    def sample(self, conditions, key_padding_mask):
        self.conditions = conditions
        self.batch_size = conditions.shape[0]
        self.key_padding_mask = key_padding_mask
        recon_x = self.p_sample_loop()
        return recon_x

    @torch.no_grad()
    def p_sample_loop(self):
        x = torch.randn_like(self.conditions)
        # x = torch.randn(self.batch_size, self.var_num, self.var_dim, device=self.device)
        for i in reversed(range(0, self.num_timesteps)):
            x = self.p_sample(x, torch.full((self.batch_size,), i, device=self.device, dtype=torch.long))
        return x

    def constraint_guided_sample(self, conditions, key_padding_mask, A, b, c):
        self.conditions = conditions
        self.batch_size = conditions.shape[0]
        self.key_padding_mask = key_padding_mask
        recon_x = self.constraint_guided_p_sample_loop(A, b, c)
        return recon_x

    def constraint_guided_p_sample_loop(self, A, b, c):
        x = torch.randn_like(self.conditions)
        # x = torch.randn(self.batch_size, self.var_num, self.var_dim, device=self.device)
        for i in reversed(range(0, self.num_timesteps)):
            x = self.constraint_guided_p_sample(x, torch.full((self.batch_size,), i, device=self.device, dtype=torch.long), A, b, c)
        return x

    def constraint_guided_p_sample(self, x, t, A, b, c, clip_denoised=True, repeat_noise=False):
        with torch.no_grad():
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        # guide graident update
        x_t = model_mean.detach()
        x_t.requires_grad = True
        pred_x, _ = self.decoder(self.conditions, x_t, self.key_padding_mask)
        pred_x = pred_x.view(self.conditions.shape[0], -1, 1)
        violates = torch.max((A.to_dense() @ pred_x).squeeze() - b, 
                            torch.tensor(0)).sum()

        obj = (pred_x.squeeze() @ c).sum()

        loss = (1-self.obj_guided_coef)*violates + self.obj_guided_coef*obj
        loss.backward()

        std = (0.5 * model_log_variance).exp()
        model_mean = model_mean - self.gradient_scale * std * x_t.grad

        # print(f"iteration: {t}, violates: {violates.item()}")

        noise = noise_like(x.shape, self.device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(self.batch_size, *((1,) * (len(x.shape) - 1)))
        # print(loss.item())
        return model_mean + nonzero_mask * std * noise

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, self.device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(self.batch_size, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, t, clip_denoised=True):
        try:
            model_out = self.predict_model.apply_model(x, t, self.conditions, self.key_padding_mask)
        except Exception as e:
            print(x.shape)
            print(x)
            print(t)
            raise Exception(e)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def get_loss(self, target, pred, loss_type):
        if loss_type == "l1":
            loss = (target - pred).abs()
            loss = torch.mean(loss)
        elif loss_type == "l2":
            loss = torch.nn.functional.mse_loss(target.squeeze(), pred.squeeze(), reduction='none')
            loss = torch.mean(loss)
        elif loss_type == "bce":
            # scale_layer = torch.nn.Sigmoid()
            # scale_pred = scale_layer(pred)
            # scale_pred = torch.clamp(pred, 0, 1)
            bce_loss_net = torch.nn.BCELoss()
            loss = bce_loss_net(pred, target)

        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

class DDIMSampler(torch.nn.Module):
    def __init__(self, trainer_model, decoder=None, gradient_scale = 1000, obj_guided_coef = 0, device = "cpu"):
        super().__init__()
        self.ddim_num_steps = None
        self.device = device
        self.model = trainer_model
        self.ddpm_num_timesteps = self.model.num_timesteps
        self.parameterization = self.model.parameterization
        self.decoder = decoder
        self.gradient_scale = gradient_scale
        self.initial_noise = None

        self.obj_guided_coef = obj_guided_coef

        self.v_posterior = 0
        self.original_elbo_weight = 0.
        self.schedule = "linear"

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        self.ddim_num_steps = self.ddim_num_steps
        betas = self.model.betas
        alphas_cumprod = self.model.alphas_cumprod
        alphas_cumprod_prev = self.model.alphas_cumprod_prev

        timesteps, = betas.shape
        self.ddpm_num_timesteps = int(timesteps)
        self.linear_start = self.model.linear_start
        self.linear_end = self.model.linear_end

        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', to_torch(ddim_sigmas))
        self.register_buffer('ddim_alphas', to_torch(ddim_alphas))
        self.register_buffer('ddim_alphas_prev', torch.from_numpy(ddim_alphas_prev))
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def constraint_guided_sample(self, conditions, key_padding_mask, A, b, c, S=100):
        self.make_schedule(ddim_num_steps=S)
        self.predict_model = self.model.predict_model
        self.batch_size = conditions.shape[0]
        self.key_padding_mask = key_padding_mask
        self.conditions = conditions
        

        x, intermediates = self.consraint_guided_ddim_sampling(S, A, b, c)
        # norm_x = torch.nn.functional.normalize(x, p=2, dim=0)
        # norm_x = (norm_x + 1)/2
        return x, intermediates
        
    def consraint_guided_ddim_sampling(self, S, A, b, c):
        x = torch.randn_like(self.conditions)
        if self.initial_noise is not None:
            x = self.initial_noise
        self.initial_noise = x
        intermediates = [x]
        for step in reversed(range(0, S)):
            index = S - step - 1
            x, pred_x0 = self.consraint_guided_p_sample_ddim(x, step, index, A, b, c)
            intermediates.append(x)
        return x, intermediates

    def consraint_guided_p_sample_ddim(self, x, t, index, A, b, c,  repeat_noise=False, temperature=1):
        with torch.no_grad():
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((self.batch_size, 1, 1), alphas[index], device=self.device)
            a_prev = torch.full((self.batch_size, 1, 1), alphas_prev[index], device=self.device)
            sigma_t = torch.full((self.batch_size, 1, 1), sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((self.batch_size, 1, 1), sqrt_one_minus_alphas[index], device=self.device)

            if self.parameterization == 'x0':
                pred_x0 = self.predict_model.apply_model(x,
                                                torch.full((self.batch_size,), t, device=self.device, dtype=torch.long),
                                                self.conditions, self.key_padding_mask)
                e_t = (x - pred_x0 * a_t.sqrt()) / sqrt_one_minus_at

            else:
                e_t = self.predict_model.apply_model(x,
                                                torch.full((self.batch_size,), t, device=self.device, dtype=torch.long),
                                                self.conditions, self.key_padding_mask)
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # guide graident update
        x_t = x.detach()
        x_t.requires_grad = True
        pred_x, _ = self.decoder(self.conditions, x_t, self.key_padding_mask)

        pred_x = pred_x.view(self.conditions.shape[0], -1, 1)
        violates = torch.max(torch_sparse.spmm(A.coalesce().indices(), A.coalesce().values(), A.size()[0], A.size()[1], pred_x).squeeze() - b, 
                            torch.tensor(0)).sum()

        obj = (pred_x.squeeze() @ c).sum()
        # violates = torch.max((A.mm(pred_x.unsqueeze(dim=1)).squeeze() - b), 
        #                     torch.tensor(0)).sum()
        
        # obj = (c*pred_x).sum()

        loss = (1-self.obj_guided_coef)*violates + self.obj_guided_coef*obj
        loss.backward()

        # print(violates.item(), obj.item())
        
        e_t = e_t - self.gradient_scale*x_t.grad
        #pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, self.device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(self, conditions, key_padding_mask, S=100):
        self.make_schedule(ddim_num_steps=S)
        self.predict_model = self.model.predict_model
        self.batch_size = conditions.shape[0]
        self.key_padding_mask = key_padding_mask

        x, intermediates = self.ddim_sampling(S, conditions)
        # norm_x = torch.nn.functional.normalize(x, p=2, dim=0)
        # norm_x = (norm_x + 1)/2
        return x, intermediates

    @torch.no_grad()
    def ddim_sampling(self, S, condition=None):
        x = torch.randn_like(condition)
        if self.initial_noise is not None:
            x = self.initial_noise
        self.initial_noise = x
        intermediates = [x]
        for step in reversed(range(0, S)):
            index = S - step - 1
            x, pred_x0 = self.p_sample_ddim(x, step, condition, index)
            intermediates.append(x)
        return x, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, t, condition, index, repeat_noise=False, temperature=1):
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        a_t = torch.full((self.batch_size, 1, 1), alphas[index], device=self.device)
        a_prev = torch.full((self.batch_size, 1, 1), alphas_prev[index], device=self.device)
        sigma_t = torch.full((self.batch_size, 1, 1), sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((self.batch_size, 1, 1), sqrt_one_minus_alphas[index], device=self.device)

        if self.parameterization == 'x0':
            pred_x0 = self.predict_model.apply_model(x,
                                             torch.full((self.batch_size,), t, device=self.device, dtype=torch.long),
                                             condition, self.key_padding_mask)
            e_t = (x - pred_x0 * a_t.sqrt()) / sqrt_one_minus_at

        else:
            e_t = self.predict_model.apply_model(x,
                                             torch.full((self.batch_size,), t, device=self.device, dtype=torch.long),
                                             condition, self.key_padding_mask)
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()


        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, self.device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def get_loss(self, target, pred, loss_type):
        if loss_type == "l1":
            loss = (target - pred).abs()
            loss = torch.mean(loss)
        elif loss_type == "l2":
            loss = torch.nn.functional.mse_loss(target.squeeze(), pred.squeeze(), reduction='none')
            loss = torch.mean(loss)
        elif loss_type == "bce":
            bce_loss_net = torch.nn.BCELoss(reduction='none')
            loss = bce_loss_net(pred, target)
            loss = torch.mean(loss)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss