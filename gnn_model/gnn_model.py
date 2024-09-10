import torch
from gnn_model.basic_module import BaseModel, PreNormLayer, SelectiveNet, BipartiteGraphConvolution, GCNMLPModule


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

class NeuralDiving(BaseModel):
    
    """
    A simple reproducation of Neural Diving https://arxiv.org/pdf/2012.13349.pdf
    """
    
    def __init__(self, emb_size = 64, cons_nfeats = 5, edge_nfeats = 1, var_nfeats = 17, gcn_mlp_layer_num = 5):
        super().__init__()
        self.gnn_model = GNNModel(emb_size, cons_nfeats, edge_nfeats, var_nfeats, gcn_mlp_layer_num= gcn_mlp_layer_num)

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1),
            torch.nn.Sigmoid()
        )
        
        self.select_net = SelectiveNet(emb_size)


    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        variable_features = self.gnn_model(constraint_features, edge_indices, edge_features, variable_features)

        output = self.output_module(variable_features).squeeze(-1) 
        select_vars = self.select_net(variable_features).squeeze(-1)
        return output, select_vars
