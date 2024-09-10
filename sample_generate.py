from pyscipopt import Model
import numpy as np
import pickle
import os
import argparse
from multiprocessing import Process, Queue
import gurobipy as gp

from feature import ObservationFunction
from environments import RootPrimalSearch as Environment 
import json
import random
import ecole


def scip_solver(filepath, params):
    m = Model()
    m.readProblem(filepath)
    m.setParams({
    #'estimation/restarts/restartpolicy': 'n',
    #'limits/nodes': '1'
    'limits/maxsol': params['max_saved_sols'],
    'limits/time': params['time_limit'],
    'limits/gap': params['gap'],
    'constraints/countsols/collect': True,
    })
    m.hideOutput(quiet = False)
    # m.setPresolve(False)
    m.optimize()
    
    var_names = []
    sols = []
    objs = []

    vars = m.getVars(transformed = True)
    
    var_names = [v.name[2:] for v in vars]

    feasible_sols = m.getSols()
    
    for sol in feasible_sols:
        sols.append([round(m.getSolVal(sol, vars[_])) if vars[_].vtype() == 'BINARY' else m.getSolVal(sol, vars[_]) for _ in range(len(var_names))])
        objs.append(m.getSolObjVal(sol))

    sols = np.array(sols, dtype=np.float32)
    objs = np.array(objs, dtype=np.float32)
    sol_data = {
        'var_names': var_names,
        'sols': sols,
        'objs': objs,
        'gap': m.getGap(),
        # 'time':m.getSolvingTime()
    }
    print(f'solving time:{m.getSolvingTime()}')
    return sol_data

    
def gurobi_solver(filepath, params):
    gp.setParam('LogToConsole', 1)
    print(f'Solving problem instance {filepath}.')
    m = gp.read(filepath)

    m.Params.PoolSolutions = params['max_saved_sols']
    m.Params.PoolSearchMode = 2
    m.Params.TimeLimit = params['time_limit']
    # m.setParam("GRB.IntParam.SolutionLimit", 500)
    m.setParam("MIPGap", params['gap'])
    if params['sub_optimal']:
        m.setParam(gp.GRB.Param.SolutionLimit, 500)
        m.setParam(gp.GRB.Param.Heuristics, 0)
        m.setParam(gp.GRB.Param.Presolve, 0)
    m.Params.Threads = 1

    m.optimize()

    sols = []
    objs = []
    solc = m.getAttr('SolCount')

    mvars = m.getVars()
    #get variable name,
    var_names = [v.varName for v in mvars]

    for n in range(solc):
        m.Params.SolutionNumber = n
        sols.append([round(v.xn) if v.VType == 'B' else v.xn for v in m.getVars()])
        objs.append(m.PoolObjVal)


    sols = np.array(sols,dtype=np.float32)
    objs = np.array(objs,dtype=np.float32)

    sol_data = {
        'var_names': var_names,
        'sols': sols,
        'objs': objs,
        'gap' : m.MIPGap
    }

    return sol_data

def solution_collector(ins_dir, sol_dir, q, params):
    while True:
        filename = q.get()
        if not filename:
            break
        ins_file = f'{ins_dir}/{filename}.mps'
        sol_file = f'{sol_dir}/{filename}.sol'
        # instance_filename = f'./instances/{dirname}/{t}/{filename}_{i}.mps.gz'
        # soltion_filename = f'./samples/{dirname}/{t}/{filename}_{i}.sol'
        sol_data = gurobi_solver(ins_file, params)
        pickle.dump(sol_data, open(sol_file,'wb'))
        
def feature_collector(ins_dir, sol_dir, q, seed, file_type = ".mps"):
    while True: 
        filename = q.get()
        if not filename:
            break
        infile = f'{ins_dir}/{filename}'
        outfile = f'{sol_dir}/{filename}'
        print(f"Collecting feature from problem {infile}")
        observation_function = ObservationFunction()
        ## Only NV disable presolve!!!!!!!!
        env = Environment(observation_function=observation_function, presolve=True)
        # observation_function.seed(seed)
        # env.seed(seed)
        initial_bound = json.load(open(f'{infile}.json', 'r'))
    
        # fetch observation feature
        observation, _, _, _, _ = env.reset(f'{infile}.{file_type}', objective_limit = initial_bound['primal_bound'])
        if observation == None:
            print(f'{filename} has not observation !!!!!!')
            continue
        # fetch variable names ordered by transformed problem
        m = env.model.as_pyscipopt()
        vars = m.getVars(transformed = True)
        var_names = [v.name[2:] for v in vars if v.isInLP()]
        int_var_indices = [ i for i in range(len(vars)) if vars[i].isInLP() and vars[i].vtype() == 'BINARY']
        
        # acquire solutions ordered by transformed problem
        solutions = pickle.load(open(f'{outfile}.sol', 'rb'))
        # assert len(var_names) == len(solutions['var_names'])
        var_sols = {}
        for i in range(len(solutions['var_names'])):
            var_sols[solutions['var_names'][i]] = solutions['sols'][:, i].reshape(-1, 1)
        s = []
        ### Uncomment to deal with dual variable (added by scip)
        # var_num = 0
        # for v in var_names:
        #     if v in var_sols.keys():
        #         s.append(var_sols[v])
        #     else:
        #         var_names[var_num] = 'du'+ v
        #         s.append(np.zeros((solutions['sols'].shape[0],1)))
        #     var_num += 1
        ###
        for v in var_names:
            s.append(var_sols[v])
        
        sols = np.concatenate(s, axis = 1)
        print(f"The instance {infile} has {sols.shape[1]} variables in transformed problem")
        obs = {
            'var_names': var_names,
            'int_var_indices': int_var_indices,
            'observation': observation,
            'objs': solutions['objs'],
            'sols': sols
        }
        pickle.dump(obs, open(f'{outfile}.obs', 'wb'))

def show_solver_details(data_size):
    for t in data_size:
        file = t
        for i in range(data_size[t]):
            ins_file = f'binary_instances/{dirname}/{file}/{filename}_{i}.mps'
            print(ins_file)
            sol_file = f'binary_samples/{dirname}/{file}/{filename}_{i}.sol'
            sol_data = gurobi_solver(ins_file, params)
            pickle.dump(sol_data, open(sol_file, 'wb'))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    ecole.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_type', type = str, default = 'SC', help='instance type')
    parser.add_argument('--collect_object', type = str, default = 'feature', help='solution or feature')
    parser.add_argument('--max_saved_sols', type = int, default = 500, help = 'The number of solutions required to store.')
    parser.add_argument('--time_limit', type = int, default = 1000, help = 'The time limit of solver')
    parser.add_argument('--gap', type = float, default = 0, help='The objective of solver')
    parser.add_argument('--sub_optimal', type=bool, default=False, help='Whether to get low quality solutions')
    parser.add_argument('--train_size', type = int, default=800, help= "The size of train samples")
    parser.add_argument('--valid_size', type = int, default=100, help= "The size of valid samples")
    parser.add_argument('--test_size', type = float, default=100, help= "The size of test samples")
    args = parser.parse_args()
    
    
    params = {}
    params['max_saved_sols'] = args.max_saved_sols
    params['time_limit'] = args.time_limit
    params['gap'] = args.gap
    params['sub_optimal'] = args.sub_optimal

    problem_type = args.problem_type

    if problem_type == "SC":
        filename = 'set_cover'
        dirname = '1_' + filename
        filetype = 'mps'
        start_valid_idx = 800
        start_test_idx = 900
    elif problem_type == "CA":
        filename = 'combinatorial_auction'
        dirname = '2_' + filename
        start_valid_idx = 800
        start_test_idx = 900
        filetype = 'mps'
    elif problem_type == "CF":
        filename = 'capacity_facility'
        dirname = '3_' + filename
        filetype = 'mps'
        start_valid_idx = 800
        start_test_idx = 900

    data_size = {'train': 1, 'valid': args.valid_size, 'test': args.test_size}

    show_solver_details(data_size)
    
    
    # n_workers = 20
    # q = Queue()
    # for t in data_size.keys():
    #     for i in range(data_size[t]):
    #         if t == 'train':
    #             q.put(f'{dirname}/{t}/{filename}_{i}')
    #         elif t == 'valid':
    #             index = start_valid_idx+i
    #             # index = i
    #             q.put(f'{dirname}/{t}/{filename}_{index}')
    #         else:
    #             index = start_test_idx+i
    #             # index = i
    #             q.put(f'{dirname}/{t}/{filename}_{index}')
    #
    # for i in range(n_workers):
    #     q.put(None)
    #
    #
    # ins_dir = f'binary_instances'
    # sol_dir = f'binary_samples'
    #
    # collect_object = args.collect_object  ## feature or solution
    #
    # # collect solutions or features
    # ps = []
    # for i in range(n_workers):
    #     if collect_object == 'solution':
    #         p = Process(target=solution_collector, args=(ins_dir, sol_dir, q, params))
    #     else:
    #         p = Process(target=feature_collector, args=(ins_dir, sol_dir, q, i, filetype))
    #     p.start()
    #     ps.append(p)
    # for p in ps:
    #     p.join()
    #
    # print('Done')
    
    
    

            
            

        
    
    

    

