import ecole
import os
import json
import gurobipy as gp
import numpy as np

def milpGenerator(file_size, problem_type):
    if problem_type == "SC":
        filename = 'set_cover'
        dirname = '1_' + filename
        instance_generator = ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=2000, density=0.05, max_coef=100)
    elif problem_type == "CA":
        filename = 'combinatorial_auction'
        dirname = '2_' + filename
        instance_generator = ecole.instance.CombinatorialAuctionGenerator(n_items=500, n_bids=1500)
    elif problem_type == "CF":
        filename = 'capacity_facility'
        dirname = '3_' + filename
        instance_generator = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=50,
                                                                                 demand_interval=(5,36),
                                                                                 continuous_assignment=False)

    index = 0
    for t in ['train', 'valid', 'test']:
        for i in range(file_size[t]):
            instance = next(instance_generator)
            problem_path = f'./new_instances/{dirname}/{t}/{filename}_{index}'
            problem = instance.copy_orig().as_pyscipopt()
            problem.writeProblem(f'{problem_path}.mps')
            pb_model = instance.copy_orig().as_pyscipopt()
            db_model = instance.copy_orig().as_pyscipopt()
            pb_model.setParam('limits/solutions', 1)
            for v in db_model.getVars():
                db_model.chgVarType(v, 'CONTINUOUS')
            pb_model.optimize()
            db_model.optimize()
            initial_bound = {}
            initial_bound['primal_bound'] = pb_model.getPrimalbound()
            initial_bound['dual_bound'] = db_model.getObjVal()
            json.dump(initial_bound, open(f'{problem_path}.json', 'w'))
            index += 1



def gurobi_solver(filepath, params):
    gp.setParam('LogToConsole', 1)
    print(f'Solving problem instance {filepath}')
    m = gp.read(f'{filepath}.mps')

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

def browser_check(dirname):
    if not os.path.isdir(f'./new_instances/{dirname}'):
        os.mkdir(f'./new_instances/{dirname}')
    if not os.path.isdir(f'./new_instances/{dirname}/train'):
        os.mkdir(f'./new_instances/{dirname}/train')
    if not os.path.isdir(f'./new_instances/{dirname}/valid'):
        os.mkdir(f'./new_instances/{dirname}/valid')
    if not os.path.isdir(f'./new_instances/{dirname}/test'):
        os.mkdir(f'./new_instances/{dirname}/test')
    if not os.path.isdir(f'./new_samples/{dirname}'):
        os.mkdir(f'./new_samples/{dirname}')
    if not os.path.isdir(f'./new_samples/{dirname}/train'):
        os.mkdir(f'./new_samples/{dirname}/train')
    if not os.path.isdir(f'./new_samples/{dirname}/valid'):
        os.mkdir(f'./new_samples/{dirname}/valid')
    if not os.path.isdir(f'./new_samples/{dirname}/test'):
        os.mkdir(f'./new_samples/{dirname}/test')


if __name__ == '__main__':
    problem_type = "SC"
    file_size = {'train': 800, 'valid': 100, 'test': 100}
    if problem_type == "SC":
        dirname = '1_set_cover'
    elif problem_type == "CA":
        dirname = '2_combinatorial_auction'
    elif problem_type == "CF":
        dirname = '3_capacity_facility'

    browser_check(dirname)

    milpGenerator(file_size, problem_type)

    # filepath = f'temp_instances/{dirname}_0'
    # solving_params = {}
    # solving_params['max_saved_sols'] = 500
    # solving_params['time_limit'] = 3600
    # solving_params['gap'] = 0
    # solving_params['sub_optimal'] = False
    # gurobi_solver(filepath=filepath, params=solving_params)