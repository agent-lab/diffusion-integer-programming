import torch

class CLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

    def forward(self, mip_features, x_features):
        mip_features = mip_features.view(mip_features.shape[0],-1)
        x = x.view(mip_features.shape[0],-1)
        
        mip_features = mip_features / mip_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_mip = logit_scale * mip_features @ x_features.t()
        logits_per_x = logits_per_mip.t()
        return logits_per_mip, logits_per_x