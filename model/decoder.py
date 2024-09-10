from collections import OrderedDict

import torch
import torch_geometric

from model.modules import ResidualAttentionBlock, QuickGELU

from model.modules import SelectiveNet

class MipConditionalDecorder(torch.nn.Module):
    def __init__(self, attn_dim: int, n_heads: int, n_layers: int, attn_mask: torch.Tensor = None, use_select_net = False):
        super().__init__()

        self.attn_dim = attn_dim

        self.resblocks = torch.nn.ModuleList([ResidualAttentionBlock(attn_dim, n_heads, attn_mask) for _ in range(n_layers)])

        self.linear_proj = torch.nn.Sequential(OrderedDict([
                    ("linear_projection", torch.nn.Linear(attn_dim, 1))
                    ]))
        
        
        self.use_select_net = use_select_net
        if self.use_select_net:
            # self.linear_proj_sel = torch.nn.Sequential(OrderedDict([
            #         ("linear_projection", torch.nn.Linear(attn_dim, 1))
                    
            #         ]))
            self.select_net = torch.nn.Sequential(
                torch.nn.Linear(attn_dim, attn_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(attn_dim, 1)
            )
        # self.ln = torch.nn.LayerNorm(attn_dim)

        # self.attn_mask = attn_mask

    def forward(self, mip_features: torch.Tensor, x_features: torch.Tensor, key_padding_mask: torch.Tensor = None):

        y = torch.concat([mip_features, x_features], dim=1)
        
        concat_key_padding_mask = key_padding_mask if key_padding_mask is None else torch.concat([key_padding_mask, key_padding_mask], dim=1)
        for module in self.resblocks:
            y = module(y, concat_key_padding_mask)
        y = y[:, -mip_features.shape[1]:, :]

        z = self.linear_proj(y).squeeze(dim=-1)
        z.masked_fill_(key_padding_mask, -torch.inf)
        z = torch.sigmoid(z)
        sols = torch.masked_select(z, ~key_padding_mask)
        selected = None
        if self.use_select_net:
            selected_z = self.select_net(y).squeeze(dim=-1)
            selected_z.masked_fill_(key_padding_mask, -torch.inf)
            selected_z = torch.sigmoid(selected_z)
            selected = torch.masked_select(selected_z, ~key_padding_mask)
        return sols, selected
    

class SelectiveNetLoss:
    def __init__(self, lamb = 32, coverage = 0.50):
        self.bce_loss = torch.nn.BCELoss(reduction= 'none')
        self.sum_aggr = torch_geometric.nn.aggr.SumAggregation()
        self.mean_aggr = torch_geometric.nn.aggr.MeanAggregation()
        self.lamb = lamb
        self.coverage = coverage
        
    def compute(self, recon_solution, selected_vars, batch):
        bce_loss = self.bce_loss(recon_solution, batch.solution[batch.int_indices].float())
        sum_yd = self.sum_aggr(selected_vars.unsqueeze(dim = 1), batch.int_vars_batch)
        mean_yd = self.mean_aggr(selected_vars.unsqueeze(dim = 1), batch.int_vars_batch)
        weights = selected_vars/torch.repeat_interleave(sum_yd.squeeze(), batch.n_int_vars, dim = 0)
        
        weighted_loss = self.sum_aggr((weights*bce_loss).unsqueeze(dim=1), batch.int_vars_batch)
        penalty_term = self.lamb*(self.coverage - mean_yd)**2
        return (weighted_loss+penalty_term).squeeze()