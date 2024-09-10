import json
from statistics import mean

import torch
from sympy.stats import Binomial

from environments import RootPrimalSearch as Environment
from feature import ObservationFunction
from utils import extract_bigraph_from_mps

from pyscipopt import Model
def get_primal_obj(filename, instance_file_type,  sample_nums):
    pb_model = Model()
    pb_model.readProblem(f"{filename}.{instance_file_type}")
    pb_model.setParam('limits/solutions', 1)
    pb_model.optimize()
    primal_bound = pb_model.getPrimalbound()
    return primal_bound

@torch.no_grad()
def get_gnn_obj(filename, instance_file_type, sample_nums, gnn_model):
    device = gnn_model.device()
    observation_function = ObservationFunction()
    env = Environment(observation_function=observation_function, presolve=True)
    bigraph = extract_bigraph_from_mps(filename, instance_file_type, observation_function, env)
    bigraph = bigraph.to(device)
    A = torch.sparse_coo_tensor(
        bigraph.edge_index,
        bigraph.edge_attr.squeeze(),
        size=(bigraph.constraint_features.shape[0], bigraph.variable_features.shape[0]))
    A = A.index_select(1, bigraph.int_indices)

    b = bigraph.constraint_features[:, 0]

    c = bigraph.variable_features[:, 0]

    feasible_num = 0
    obj_vals = []
    model = env.model.as_pyscipopt()
    for n in range(sample_nums):
        output, select = model(
            bigraph.constraint_features,
            bigraph.edge_index,
            bigraph.edge_attr,
            bigraph.variable_features
        )
        p = Binomial(1, select)
        probs = p.sample() * output
        p = Binomial(1, output[probs > 0])
        # action set主要是位置
        action_set = list(range(len(probs)))
        action = (action_set[probs > 0].cpu(), p.sample().cpu())

        s = model.createSol()
        j = 0
        for v in model.getVars(transformed=True):
            model.setSolVal(s, v, action[j].item())
            j += 1
        obj_val = model.getSolObjVal(s)
        obj_vals.append(obj_val)

    print(obj_vals)





if __name__ =="__main__":
    instance = "SC"
    if instance == 'SC':
        instance_file = '1_set_cover'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
    elif instance == 'CA':
        instance_file = '2_combinatorial_auction'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'max'
    elif instance == 'CF':
        instance_file = '3_capacity_facility'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
    elif instance == 'IS':
        instance_file = '4_independent_set'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'max'
    primals = []
    for i in range(10):
        file_path = f"binary_instances/{instance_file}/test/{instance_file[2:]}_{start + i}.json"
        with open(file_path, 'r') as file:
            data = json.load(file)
            primals.append(data["primal_bound"])
    print(mean(primals))
