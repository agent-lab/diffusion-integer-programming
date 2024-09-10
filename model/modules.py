from collections import OrderedDict

import torch
import numpy as np
import torch_geometric

"""
This file includes basic model for solving mixed integer programming. Note that PreNormLayer is still unused.
"""

class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class SelectiveNet(torch.nn.Module):
    
    """
    Selective network to decide whether variables need to be predicted.
    """
    
    def __init__(self, feature_size):
        super().__init__()
        self.select_module = torch.nn.Sequential(
            torch.nn.Linear(feature_size, feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, 1, bias=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, feature):
        return self.select_module(feature)
    
class GCNMLPModule(torch.nn.Module):
    """
    Reproducde Z = Af(U) and Z_tilde in Neural Diving (Nair et al. 2019)
    """
    def __init__(self, cons_input_size = 64, vars_input_size = 64, output_size = 64):
        super().__init__()
        
        self.cons_mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(cons_input_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.ReLU()
        )
        self.vars_mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(vars_input_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.ReLU()
        )
        self.jump_vars_mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(output_size*2, output_size),
            torch.nn.Sigmoid()
        )
        
        self.jump_cons_mlp_layer = torch.nn.Sequential(
            torch.nn.Linear(output_size*2, output_size),
            torch.nn.Sigmoid()
        )
        
        self.vars_layer_norm = torch.nn.LayerNorm(output_size)
        self.cons_layer_norm = torch.nn.LayerNorm(output_size)
        
    def jump_network_forward(self, next_z, z_tilde):
        next_cons_fea, next_vars_fea = next_z
        tilde_cons_fea, tilde_vars_fea = z_tilde
        
        # concat
        next_tilde_cons_fea = self.jump_cons_mlp_layer(torch.concat([next_cons_fea, tilde_cons_fea], dim = 1))
        next_tilde_vars_fea = self.jump_vars_mlp_layer(torch.concat([next_vars_fea, tilde_vars_fea], dim = 1))
        
        next_z_tilde = (next_tilde_cons_fea, next_tilde_vars_fea)
        return next_z_tilde
        
    def forward(self, z, z_tilde, edge_indices, edge_attrs):
        constraint_features, variable_features = z
        A = torch.sparse_coo_tensor(
            edge_indices,
            edge_attrs.squeeze(),
            size=(constraint_features.shape[0], variable_features.shape[0])
        ).detach()
        A_T = A.transpose(0,1).detach()
        mlp_cons_fea = self.cons_mlp_layer(constraint_features)
        mlp_vars_fea = self.vars_mlp_layer(variable_features)
        
        # graph convolution Z = Af(Z)
        gc_cons_fea = A.mm(mlp_vars_fea) + constraint_features
        gc_vars_fea = A_T.mm(mlp_cons_fea) + variable_features
        
        
        # layer normalization
        ln_cons_fea = self.cons_layer_norm(gc_cons_fea)
        ln_vars_fea = self.vars_layer_norm(gc_vars_fea)
        
        # jump network
        next_z = (ln_cons_fea, ln_vars_fea)
        next_z_tilde = self.jump_network_forward(next_z, z_tilde)

        return next_z, next_z_tilde
    
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size = 64):
        super().__init__('add')

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output

class GNNModel(torch.nn.Module):
    def __init__(self, emb_size = 64, cons_nfeats = 5, edge_nfeats = 1, var_nfeats = 17, gcn_mlp_layer_num = 5):
        super().__init__()
        self.gcn_mlp_layer_num = gcn_mlp_layer_num
        
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)
        
        self.gcn_mlp_layer = torch.nn.ModuleList([GCNMLPModule(emb_size, emb_size, emb_size) for _ in range(gcn_mlp_layer_num)])



    def forward(self, constraint_features, edge_indices, edge_attrs, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_attrs)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features) 
        
        z = (constraint_features, variable_features)
        z_tlide = (constraint_features, variable_features)
        for module in self.gcn_mlp_layer:
            next_z, next_z_tlide = module(z, z_tlide, edge_indices, edge_attrs)
            z = next_z
            z_tlide = next_z_tlide
        
        constraint_features, variable_features = z_tlide
        
        return variable_features
    
class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()


        self.attn = torch.nn.MultiheadAttention(d_model, n_head, batch_first = True)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(d_model, d_model)),
            ("gelu", QuickGELU()),
            ("c_proj", torch.nn.Linear(d_model, d_model))
        ]))
        self.ln_2 =  torch.nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x