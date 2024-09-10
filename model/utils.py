import torch
import numpy as np
import torch_geometric

from model.encoder import MIPEncoder



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prenorm(model, pretrain_loader):
    """
    Pre-nomalize all PreNorm layers in the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to pre-normalize.
    pretrain_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of pre-training samples.

    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    model.pre_norm_init()
    
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            pre_norm = True
            if isinstance(model, MIPEncoder):
                pre_norm = model.pre_norm(
                    batch.constraint_features, 
                    batch.edge_index, 
                    batch.edge_attr, 
                    batch.variable_features
                )
            # elif isinstance(model, NeuralDiving):
            #     pre_norm = model.pre_norm(
            #         batch.constraint_features, 
            #         batch.edge_index, 
            #         batch.edge_attr, 
            #         batch.variable_features
            #     )
            # elif isinstance(model, ConditionalVAE):
            #     pre_norm =  model.pre_norm(
            #         batch.constraint_features,
            #         batch.edge_index,
            #         batch.edge_attr,
            #         batch.variable_features,
            #         batch.vars_batch,
            #         batch.n_vars, 
            #         batch.int_vars_batch,
            #         batch.n_int_vars,
            #         batch.int_indices,
            #         batch.solution
            #     )
            
            if not pre_norm:
                break

        if  model.pre_norm_next() is None:
            break
        i += 1
    return i

## get padding
# def get_mask_index(n_batches, total_len, int_vars_batch):
#     one_hot = torch.zeros((n_batches, total_len)).to(int_vars_batch.device)
#     one_hot.scatter_(0, int_vars_batch.unsqueeze(1).T, 1)
#     mask_index = one_hot.bool()
#     return mask_index

# def get_padding_x(x, n_batches, int_vars_batch, padding_len):
#     mask_index = get_mask_index(n_batches, x.shape[0], int_vars_batch)
#     padding_sol_list = []
#     for i in range(n_batches):
#         solution = x[mask_index[i]]
#         padding_sol_list.append(torch.nn.functional.pad(solution, (0, padding_len-solution.shape[0]), 'constant', 2).unsqueeze(0))
#     padding_sols = torch.concat(padding_sol_list, dim=0)
#     key_padding_mask = (padding_sols==2)
#     return padding_sols, key_padding_mask

# def get_padding_mip(mip_features, n_batches, int_vars_batch, padding_len):
#     mask_index = get_mask_index(n_batches, mip_features.shape[0], int_vars_batch)
#     padding_mip_feature_list = []
#     for i in range(n_batches):
#         mip_feature = mip_features[mask_index[i]]
#         padding_mip_feature_list.append(torch.nn.functional.pad(mip_feature, (0, 0, 0, padding_len-mip_feature.shape[0]), 'constant', 0).unsqueeze(0))
#     mip_features = torch.concat(padding_mip_feature_list, dim=0)
#     return mip_features


def get_padding(features, n_int_vars, padding_len, pad_object):
    if isinstance(n_int_vars, int):
        split_features = torch.split(features, [n_int_vars])
    else:
        split_features = torch.split(features, n_int_vars.tolist())
    if pad_object == 'mip':
        padding_features = map(lambda v: torch.nn.functional.pad(v, (0, 0, 0, padding_len-v.shape[0]), 'constant', 0).unsqueeze(0), split_features)
    elif pad_object == 'solution':
        padding_features = map(lambda v: torch.nn.functional.pad(v, (0, padding_len-v.shape[0]), 'constant', 2).unsqueeze(0), split_features)
    else:
        raise Exception("Padding object not found!")
    key_padding_mask = map(lambda v:torch.nn.functional.pad(
        torch.zeros(v.shape[0], dtype= torch.bool, device = v.device), (0, padding_len-v.shape[0]), 'constant', True).unsqueeze(0), 
        split_features)

    features_out = torch.concat(list(padding_features), dim=0)
    key_padding_mask = torch.concat(list(key_padding_mask), dim=0)
    return features_out, key_padding_mask

# def get_padding_x_v2(x, n_int_vars, padding_len):
#     split_x = torch.split(x, mip.n_int_vars.tolist())
#     padding_mip = map(lambda v: torch.nn.functional.pad(mip_feature, (0, 0, 0, padding_len-v.shape[0]).unsqueeze(0), 'constant', 0), split_mip_features)
#     mip_feature = torch.concat(list(padding_mip), dim=0)


## diffusion 
def isfunction(obj):
    return callable(obj)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    """
    collect the data in a at the index of t
    a:[time_step]
    t:[batch_size]
    return:[batch_size, 1]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def to_torch(x):
    if torch.is_tensor(x):
        out = x.clone().detach().float()
    else:
        out = torch.from_numpy(x).float()
    return out


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    give the sequence of betas
    return: tensor(n_timestep)
    """
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev