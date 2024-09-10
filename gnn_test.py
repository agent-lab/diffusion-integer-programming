from statistics import mean

from torch.distributions import Binomial
from pyscipopt import Model
from gnn_model.gnn_model import NeuralDiving
import torch
import argparse
import numpy as np
import random
from utils import extract_bigraph_from_mps

from feature import ObservationFunction
from environments import RootPrimalSearch as Environment
import json
from pyscipopt import SCIP_PARAMSETTING
import pickle

import time

def is_feasible(log_path):
    log_file = open(log_path, "r", encoding='UTF-8')
    lines = log_file.readlines()  # 读取文件的所有行
    assert len(lines) >= 0, '文件行数不足'
    second_line = lines[1].split()  # 获取第二行内容
    if "failed" in second_line:  # 判断第二行是否包含"fail"单词
        return False
    else:
        return float(second_line[-1])

def get_primal_obj(filename, instance_file_type):
    pb_model = Model()
    pb_model.readProblem(f"{filename}.{instance_file_type}")
    pb_model.setParam('limits/solutions', 1)
    pb_model.optimize()
    primal_bound = pb_model.getPrimalbound()
    return primal_bound

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--instance', type=str, default='SC', help='The instance for testing MIP, SC or CA, CF or IS')
    parser.add_argument('--test_size', type=int, default=100, help='The number of instances')
    parser.add_argument('--sample_nums', type=int, default=30, help='The number of instances')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding size in gnn')
    parser.add_argument('--gcn_layer_num', type=int, default=2, help='The number of layer in gnn')

    parser.add_argument('--time_limit', type=int, default=1.1, help='The time of scip solving')
    parser.add_argument('--coverage', type=float, default=0.2, help='')
    parser.add_argument('--output_type', type=str, default='save', help='print or save')

    args = parser.parse_args()


    if args.instance == 'SC':
        instance_file = '1_set_cover'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
    elif args.instance == 'CA':
        instance_file = '2_combinatorial_auction'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'max'
    elif args.instance == 'CF':
        instance_file = '3_capacity_facility'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
    elif args.instance == 'IS':
        instance_file = '4_independent_set'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'max'
        var_nums = 1500
    elif args.instance == 'SC_3000':
        args.instance = 'SC'
        instance_file = 'SC_3000'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'min'
    elif args.instance == 'SC_4000':
        args.instance = 'SC'
        instance_file = 'SC_4000'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'min'

    test_files = []
    for i in range(args.test_size):
        ## test_files.append(f'instances/{instance_file}/test/{instance_file[2:]}_{start + i}')
        test_files.append(f'instances/{instance_file}/set_cover_{start + i}')
    observation_function = ObservationFunction()
    env = Environment(observation_function=observation_function, presolve=True)
    gnn_model = NeuralDiving(emb_size=args.embedding_size, gcn_mlp_layer_num=args.gcn_layer_num).to(device)
    gnn_model.load_state_dict(torch.load(f'./gnn_model_hub/useSelectiveNet-True_C{args.coverage*100}_{args.instance}_Neural Diving_model_99.pkl', map_location=device))

    times = []

    obj_values = []
    for i, filename in enumerate(test_files):
        ## extract  feature

        start = time.time()
        bigraph = extract_bigraph_from_mps(filename, instance_file_type, observation_function, env)

        bigraph = bigraph.to(device)
        A = torch.sparse_coo_tensor(
            bigraph.edge_index,
            bigraph.edge_attr.squeeze(),
            size=(bigraph.constraint_features.shape[0], bigraph.variable_features.shape[0]))
        A = A.index_select(1, bigraph.int_indices)

        b = bigraph.constraint_features[:, 0]

        c = bigraph.variable_features[:, 0]

        # get predict solution
        gnn_model.eval()
        mid0 = time.time()
        output, select = gnn_model(
            bigraph.constraint_features,
            bigraph.edge_index,
            bigraph.edge_attr,
            bigraph.variable_features
        )

        initial_bound = json.load(open(f'{filename}.json', 'rb'))
        mid1 = time.time()
        observation_function = ObservationFunction()
        env = Environment(time_limit=args.time_limit, observation_function=observation_function)
        observation, action_set, reward, done, info = env.reset(filename + '.' + instance_file_type, \
                                                                objective_limit=initial_bound['primal_bound'])
        action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(device)
        m = env.model.as_pyscipopt()
        output = output[action_set]
        select = select[action_set]
        mid2 = time.time()
        # feasible_num = 0
        # obj_values = []
        problem_obj = []
        for n in range(args.sample_nums):
            p = Binomial(1, select)
            probs = p.sample() * output
            p = Binomial(1, output[probs > 0])
            action = (action_set[probs > 0].cpu(), p.sample().cpu())

            scip_model = Model()
            # add the collected partial solutions to scip and optimize
            scip_model.setParam('limits/time', args.time_limit)
            scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
            scip_model.setParam('heuristics/completesol/solutions', 5)
            # scip_model.setParam('limits/solutions', 5)
            scip_model.setObjlimit(initial_bound['primal_bound'])
            # scip_model.setParam('estimation/restarts/restartpolicy', 'n')
            # scip_model.setParam('limits/maxorigsol', 1)
            scip_model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
            log_path = f'GenMIP/agents/time_logs/neural_diving/{instance_file}/{args.instance}_gnn_{args.coverage}_instance{i}_num{n}.log'
            scip_model.setLogfile(log_path)
            scip_model.readProblem(filename + '.' + instance_file_type)

            name_var_dict = {}
            vars = m.getVars(transformed=True)

            for v in scip_model.getVars(transformed=True):
                name_var_dict[v.name] = v

            s = scip_model.createPartialSol()
            for j in range(len(action[0])):
                scip_model.setSolVal(s, name_var_dict[vars[action[0][j]].name[2:]], action[1][j])
            scip_model.addSol(s)

            scip_model.hideOutput(quiet=True)
            scip_model.optimize()

            result = is_feasible(log_path)

            if result != False:
                problem_obj.append(result)
            else:
                problem_obj.append(np.nan)
        ##problem_values = np.array(problem_values)
        end = time.time()

        

        nan_count = np.isnan(problem_obj).sum()
        print(f"instance {i}, feasible ratio: {(args.sample_nums-nan_count)/args.sample_nums}, obj value: {np.nanmean(problem_obj)}")
        print(f"total time for 30 solutions:{mid1-start + end-mid2}, average time for each solution:{(mid1-start + end-mid2)/30}, average sample time: {(mid1-mid0)/30}")
        obj_values.append(problem_obj)
        times.append([mid0-start, mid1-mid0, end - mid2])
        
    
    obj_values = np.array(obj_values)
    time_values = np.array(times)
    total_num = args.sample_nums * args.test_size
    nan_count = np.isnan(obj_values).sum()
    print(f"mean feasible ratio: {(total_num - nan_count)/total_num}, mean obj value: {np.nanmean(obj_values)}")
    
    if args.output_type == 'save':
        pickle.dump(obj_values, open(f'GenMIP/agents/time_results/neural_diving/{instance_file}/{args.instance}_gnn_{args.coverage}_obj.pkl', 'wb'))
        pickle.dump(time_values, open(f'GenMIP/agents/time_results/neural_diving/{instance_file}/{args.instance}_gnn_{args.coverage}_time.pkl', 'wb'))
       









