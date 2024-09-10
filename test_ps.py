import torch
import numpy as np
import json
import argparse
from pyscipopt import Model
from torch.distributions.binomial import Binomial
from gnn_model.gnn_model import NeuralDiving
from environments import RootPrimalSearch as Environment 
from feature import ObservationFunction
import random
# from utils import computePrimalGaps
from pyscipopt import SCIP_PARAMSETTING
import gurobipy as gp
import time

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = 'Neural Diving', help = 'The type of the trained model')
    parser.add_argument('--model_file', type = str, default = 'useSelectiveNet-False_C100_SC_Neural Diving_model_99', help = 'The file name of the trained model')
    parser.add_argument('--time_limit', type = int, default = 100, help = 'The limit time for solving MIP')
    parser.add_argument('--instance', type = str, default = 'SC', help = 'The instance for testing MIP')
    parser.add_argument('--num_instances', type = int, default = 100, help = 'The number of instances')
    parser.add_argument('--num_partial_sols', type = int, default = 1, help = 'The number of partial solutions')
    parser.add_argument('--embedding_size', type = int, default = 128, help = '')
    parser.add_argument('--gnn_fea_size', type = int, default = 128, help = '')
    parser.add_argument('--latent_size', type = int, default = 128, help = '')
    parser.add_argument('--gcn_layer_num', type = int, default = 2, help = '')
    parser.add_argument('--partial_method', type = str, default = 'selectiveNet', help='How to acquire partial solution')
    parser.add_argument('--coverage', type=float, default=0.2, help='')
    args = parser.parse_args()
    
    # choose model
    if args.model_type == 'Neural Diving':
        model = NeuralDiving(emb_size = args.embedding_size, gcn_mlp_layer_num = args.gcn_layer_num).to(device)
        model.load_state_dict(torch.load(f'./gnn_model_hub/{args.model_file}.pkl', map_location= device))

    k0 = 200
    k1 = 200
    delta = 400
    
    size = None
    
    # choose problem instance
    if args.instance == 'SC':
        instance_file = '1_set_cover'
        start = 900
        instance_file_type = '.mps'
        if size != None:
            instance_file = f'1_set_cover_{size}'
            start = 0
    elif args.instance == 'CA':
        instance_file = '2_combinatorial_auction'
        start = 900
        instance_file_type = '.mps'
    elif args.instance == 'CF':
        instance_file = '3_capacity_facility'
        start = 900
        instance_file_type = '.mps'
    elif args.instance == 'IS':
        instance_file = '4_independent_set'
        start = 0
        instance_file_type = '.mps'
    instances = []
    for i in range(start, start+args.num_instances):
        instances.append(f'{instance_file}/test/{instance_file[2:]}_{i}')
        ## instances.append(f'{instance_file}/{instance_file[2:-5]}_{i}')
    
    # gaps = np.zeros((args.num_instances, 10*args.time_limit+1))
    bounds = np.zeros((args.num_instances, 10*args.time_limit+1))
    mean = 0
    for i, instance in enumerate(instances):
       
        instance_file_name = './instances/'+ instance
        sample_file_name = './samples/'+ instance
        # create environment, get observation and milp info
        print(f"Testing {instance_file_name}......")
        initial_bound = json.load(open(f'{instance_file_name}.json', 'rb'))
        scip_model = Model()
        observation_function = ObservationFunction()
        env = Environment(time_limit = args.time_limit,
                          observation_function = observation_function)
        
        if args.model_type != 'scip':
            observation, action_set, reward, done, info = env.reset(instance_file_name+instance_file_type, \
                objective_limit = initial_bound['primal_bound'])
            
            actions = []
            num_lps = info['nlps']
            num_sols = 0
            m = env.model.as_pyscipopt()
            
            # get necessary feature from obsevation
            variable_features = observation.variable_features
            variable_features = np.delete(variable_features, 14, axis=1)
            variable_features = np.delete(variable_features, 13, axis=1)

            constraint_features = torch.FloatTensor(observation.row_features).to(device)
            edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(device)
            edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1)).to(device)
            variable_features = torch.FloatTensor(variable_features).to(device)
            action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(device)
            vars_batch = torch.zeros(variable_features.shape[0], dtype = torch.int64).to(device)
            n_vars = variable_features.shape[0]
                
            # model evalution and sample action (partial solution)
            model.eval()
            start = time.time()
            with torch.no_grad():
                
                output, select = model(
                                constraint_features,
                                edge_index,
                                edge_attr,
                                variable_features
                                )
                # output = output[action_set]
                # choose partial solution method
            
            _, indices = output.sort()
            I0 = indices[:k0]
            I1 = indices[-k1:]
            name_var_dict = {}
            vars = m.getVars(transformed = True)
            I0_name = {vars[i].name[2:] for i in I0}
            I1_name = {vars[i].name[2:] for i in I1}
            
            gp.setParam('LogToConsole', 0)
            gp_model = gp.read(instance_file_name+instance_file_type)
            gp_model.setParam(gp.GRB.Param.LogFile, f"./logs/ps_gurobi_logs/{instance_file}/{instance_file[2:]}_{start+i}.log")
            gp_model.setParam('NodeLimit', 0)
            gp_model.Params.TimeLimit = 1.5
            gp_model.Params.Threads = 1
            name_v_dict = {}
            for v in gp_model.getVars():
                name_v_dict[v.VarName] = v

            tmp_vars = []
            for name in I0_name:
                tar_var = name_v_dict[name]
                tmp_var = gp_model.addVar(name=f'alp_{tar_var}', vtype=gp.GRB.CONTINUOUS)
                tmp_vars.append(tmp_var)
                gp_model.addConstr(tmp_var >= tar_var, name= f'alpha_up_{name}')
            for name in I1_name:
                tar_var = name_v_dict[name]
                tmp_var = gp_model.addVar(name=f'alp_{tar_var}', vtype=gp.GRB.CONTINUOUS)
                tmp_vars.append(tmp_var)
                gp_model.addConstr(tmp_var >= 1- tar_var, name= f'alpha_down_{name}')
            
            all_tmp = 0
            for tmp in tmp_vars:
                all_tmp += tmp
            gp_model.addConstr(all_tmp <= delta, name="sum_tmp")
            gp_model.optimize()
            end = time.time()
            
            mean += end - start
        else:
            scip_model.setParam('limits/time', args.time_limit)
            scip_model.setObjlimit(initial_bound['primal_bound'])
            scip_model.hideOutput(quiet = True)
            # scip_model.setParam('limits/maxsol', 1000)
            scip_model.setParam('limits/maxorigsol', args.num_partial_sols)
            # scip_model.setParam('estimation/restarts/restartpolicy', 'n')
            # scip_model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
            scip_model.setLogfile(f'./logs/{instance}_{args.model_type}_{args.partial_method}.log')
            scip_model.readProblem(instance_file_name+instance_file_type)
            scip_model.optimize()
            
    print(mean/100)
            
        
    #     ## analyze result
    #     # get optimal objective value from test file
    #     sol = pickle.load(open(f'{sample_file_name}.sol', 'rb'))
    #     best_obj = sol['objs'][0]
    #     # primal_gap = {}
        
    #     primal_bound = {}
    #     # primal_gap[0] = computePrimalGaps(initial_bound['primal_bound'], best_obj)
    #     # primal_gap[0] = np.abs(initial_bound['primal_bound']-best_obj)/(np.abs(best_obj)+1e-10)
    #     primal_bound[0] = initial_bound['primal_bound']
        
    #     for line in open(f'./newlogs/{instance}_{args.model_type}.log', "r", encoding= 'UTF-8'):
    #         if '|' not in line:
    #             continue
    #         l = line.split("|")
    #         if l[0][1:-1] != 'time':
    #             # primal_gap[float(l[0][1:-1])] = np.abs(float(l[-3].strip())-best_obj)/(np.abs(best_obj)+1e-10)
    #             # primal_gap[float(l[0][1:-1])] = computePrimalGaps(float(l[-3].strip()), best_obj)
    #             if l[-3].strip()[-1] != '*':
    #                 primal_bound[float(l[0][1:-1])] = float(l[-3].strip())
    #             else:
    #                 primal_bound[float(l[0][1:-1])] = float(l[-3].strip()[0:-1])
    #     # for j in range(10*args.time_limit+1):
    #     #     if j/10 in primal_gap.keys():
    #     #         gaps[i, j] = primal_gap[j/10]
    #     #     else:
    #     #         gaps[i, j] = gaps[i, j-1]
    #     # print(f'optimal gap:{gaps[i, -1]}')
    #     for j in range(10*args.time_limit+1):
    #         if j/10 in primal_bound.keys():
    #             bounds[i, j] = primal_bound[j/10]
    #         else:
    #             bounds[i, j] = bounds[i, j-1]
    #     print(f'optimal bound:{bounds[i, -1]}')
    
    # ## plot
    # # mean_gaps = gaps.mean(axis = 0)
    # pickle.dump(bounds, open(f'./newlogs/{instance_file[2:]}_{args.model_type}.bd', 'wb'))
            
            
            

