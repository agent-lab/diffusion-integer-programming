import torch
import torch_geometric
import numpy as np
import pickle

class BipartiteNodeDataWithoutSolution(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        n_vars,
        vars_batch,
        int_indices,
        n_int_vars,
        int_vars_batch
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.n_vars = n_vars
        self.vars_batch = vars_batch
        
        self.int_indices = int_indices
        self.n_int_vars = n_int_vars
        self.int_vars_batch = int_vars_batch
        

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, solutions) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "vars_batch":
            return 1
        elif key == "int_indices":
            return self.variable_features.shape[0]
        elif key == "int_vars_batch":
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, item, *args, **kwargs):
            return super().__cat_dim__(key, item)

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        n_vars,
        solution,
        vars_batch,
        int_indices,
        n_int_vars,
        int_vars_batch
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.n_vars = n_vars
        self.solution = solution
        self.vars_batch = vars_batch
        
        self.int_indices = int_indices
        self.n_int_vars = n_int_vars
        self.int_vars_batch = int_vars_batch
        

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, solutions) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "vars_batch":
            return 1
        elif key == "int_indices":
            return self.variable_features.shape[0]
        elif key == "int_vars_batch":
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, item, *args, **kwargs):
        if key == 'solution':
            return 0
        else:
            return super().__cat_dim__(key, item)
        
class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, problem_type = 'min', position_emb= False):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.problem_type = problem_type
        self.position_emb = position_emb

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)
        
        sample_observation = sample['observation']
        sample_objs = sample['objs']
        sample_sols = sample['sols']
        
        int_indices = sample['int_var_indices']
        objs_norm = (sample_objs-sample_objs.mean())/(sample_objs.std()+1e-6)
        if self.problem_type == 'max':
            p = np.exp(objs_norm)/np.sum(np.exp(objs_norm))
        else:
            p = np.exp(-objs_norm)/np.sum(np.exp(-objs_norm))
            
        random_index = np.random.choice(range(sample_sols.shape[0]), size = 1, replace=True, p=p)
        sol = sample_sols[random_index]
        
        variable_features = sample_observation.variable_features
        variable_features = np.delete(variable_features, 14, axis=1)
        variable_features = np.delete(variable_features, 13, axis=1)

        if self.position_emb:
            lens = variable_features.shape[0]
            feature_widh = 12  # max length 4095
            position = np.arange(0, lens, 1)

            position_feature = np.zeros((lens, feature_widh))
            for i in range(len(position_feature)):
                binary = str(bin(position[i]).replace('0b', ''))

                for j in range(len(binary)):
                    position_feature[i][j] = int(binary[-(j + 1)])
            variable_features = np.concatenate([variable_features, position_feature], axis=-1)

        constraint_features = torch.FloatTensor(sample_observation.row_features)
        edge_index = torch.LongTensor(sample_observation.edge_features.indices.astype(np.int32))
        edge_attr = torch.FloatTensor(np.expand_dims(sample_observation.edge_features.values, axis=-1))
        variable_features = torch.FloatTensor(variable_features)
        n_vars = variable_features.shape[0]
        sol = torch.LongTensor(sol.astype(np.int32)).view(-1)
        int_indices = torch.LongTensor(int_indices)
        
        n_int_vars = int_indices.shape[0]
        
        
        graph = BipartiteNodeData(
            constraint_features,
            edge_index,
            edge_attr,
            variable_features,
            n_vars,
            sol,
            torch.zeros(sol.shape[0], dtype=torch.int64),
            int_indices,
            n_int_vars,
            torch.zeros(int_indices.shape[0], dtype=torch.int64)
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        return graph
        
# if __name__ == '__main__':
#     with open('./samples/4_independent_set/train/independent_set_0.obs','rb') as f:
#         fea = pickle.load(f)

#     sample_observation = fea['observation']
#     constraint_features = sample_observation.row_features
#     edge_indices = sample_observation.edge_features.indices.astype(np.int32)
#     edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
#     variable_features = sample_observation.variable_features

#     solutions1 = np.random.randn(333, 1500)
#     solutions2 = np.random.randn(234, 1500)
    
#     data1 = BipartiteNodeData(torch.FloatTensor(constraint_features), torch.LongTensor(edge_indices), torch.FloatTensor(edge_features), torch.FloatTensor(variable_features), torch.FloatTensor(solutions1), solutions1.shape[0])
#     data1.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
#     data2 = BipartiteNodeData(torch.FloatTensor(constraint_features), torch.LongTensor(edge_indices), torch.FloatTensor(edge_features), torch.FloatTensor(variable_features), torch.FloatTensor(solutions2), solutions1.shape[0])
#     data2.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
#     data_list = [data1, data2]
#     loader = torch_geometric.loader.DataLoader(data_list, batch_size = 2)
#     batch = next(iter(loader))