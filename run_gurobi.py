import gurobipy as gp
from functools import partial
from multiprocessing import Pool
from pyscipopt import Model
import time


def gurobipy_solving(instance, test_size):
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
    elif instance == 'SC_3000':
        instance_file = '1_set_cover_3000'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'min'
    elif instance == 'SC_4000':
        instance_file = '1_set_cover_4000'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'min'
        
        
    test_files = []
    for i in range(test_size):
        # test_files.append(f'./instances/{instance_file}/test/{instance_file[2:]}_{start+i}')
         test_files.append(f'./instances/{instance_file}/{instance_file[2:-5]}_{start+i}')
    cnt = 0
    mean = 0
    for filepath in test_files:    
        gp.setParam('LogToConsole', 0)
        t1 = time.time()
        m = gp.read(filepath+'.'+instance_file_type)
        m.setParam(gp.GRB.Param.LogFile, f"./GenMIP/agents/time_logs/gurobi/{instance_file}/{instance_file[2:]}_{start+cnt}.log")
        m.setParam('NodeLimit', 0)
        m.Params.TimeLimit = 10
        m.Params.Threads = 1
        m.optimize()
        t2 = time.time()
        mean += (t2-t1)
        cnt+=1
    mean /= 100
    print(mean)

def pyscipopt_solving(instance, test_size):
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
        
    test_files = []
    for i in range(test_size):
        test_files.append(f'./instances/{instance_file}/test/{instance_file[2:]}_{start+i}')
        # test_files.append(f'./instances/{instance_file}/{instance_file[2:]}_{start+i}')
    cnt = 0
    for filepath in test_files:    
        scip_model = Model()
        scip_model.setParam('limits/time', 100)
        scip_model.hideOutput(quiet = True)
        scip_model.setLogfile(f"./logs/scip_logs/{instance_file}/{instance_file[2:]}_{start+cnt}.log")
        scip_model.readProblem(filepath+'.'+instance_file_type)
        scip_model.optimize()
        cnt+=1

if __name__ == "__main__":
    processes = []
    instances = ['SC_4000']
    test_size = 100
    solver = 'gurobi'
    if solver == 'gurobi':
        with Pool() as pool:
            pool.starmap(gurobipy_solving, [(instance, test_size) for instance in instances])
    else:
        with Pool() as pool:
            pool.starmap(pyscipopt_solving, [(instance, test_size) for instance in instances])
    
    