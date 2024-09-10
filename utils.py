import numpy as np
import torch
import json
import torch_geometric

from dataset import BipartiteNodeDataWithoutSolution

def is_feasible(log_path):
    log_file = open(log_path, "r", encoding='UTF-8')
    lines = log_file.readlines()  # 读取文件的所有行
    assert len(lines) >= 0, '文件行数不足'
    second_line = lines[1].split()  # 获取第二行内容
    assert "completesol" or "partial" in second_line, '启发式异常'
    if "failed" in second_line:  # 判断第二行是否包含"fail"单词
        return False
    else:
        return float(second_line[-1])
        
def extract_bigraph_from_mps(filename, filetype, observation_function, environment):
    print(f"Collecting feature from problem {filename}")
    
    initial_bound = json.load(open(f'{filename}.json', 'r'))

    observation, _, _, _, _ = environment.reset(f'{filename}.{filetype}', objective_limit = initial_bound['primal_bound'])
    if observation == None:
        print(f'Warnning: {filename} has not observation !!!!!!')
        return None
    # fetch variable names ordered by transformed problem
    m = environment.model.as_pyscipopt()
    vars = m.getVars(transformed = True)
    var_names = [v.name[2:] for v in vars if v.isInLP()]
    int_var_indices = [ i for i in range(len(vars)) if vars[i].isInLP() and vars[i].vtype() == 'BINARY']

    sample = {
        'var_names': var_names,
        'int_var_indices': int_var_indices,
        'observation': observation,
    }

    sample_observation = sample['observation']
    
        
    int_indices = sample['int_var_indices']
        
    variable_features = sample_observation.variable_features
    # delete unnecessary features
    variable_features = np.delete(variable_features, 14, axis=1)
    variable_features = np.delete(variable_features, 13, axis=1)

    constraint_features = torch.FloatTensor(sample_observation.row_features)
    edge_index = torch.LongTensor(sample_observation.edge_features.indices.astype(np.int32))
    edge_attr = torch.FloatTensor(np.expand_dims(sample_observation.edge_features.values, axis=-1))
    variable_features = torch.FloatTensor(variable_features)
    n_vars = variable_features.shape[0]
    int_indices = torch.LongTensor(int_indices)
        
    n_int_vars = int_indices.shape[0]
        
        
    graph = BipartiteNodeDataWithoutSolution(
        constraint_features,
        edge_index,
        edge_attr,
        variable_features,
        n_vars,
        torch.zeros(variable_features.shape[0], dtype=torch.int64),
        int_indices,
        n_int_vars,
        torch.zeros(int_indices.shape[0], dtype=torch.int64)
    )

    graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
    return graph

class SelectiveNetLoss():
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