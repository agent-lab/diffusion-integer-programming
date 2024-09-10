import pickle
import numpy as np
from matplotlib import pyplot as plt
from pyscipopt import Model
from utils import is_feasible
import random

sols_path = f"./GenMIP/agents/diffusion_results/distribution_IS_ddim_20000_0.5_sols.npy"
objs_path = f"./GenMIP/agents/diffusion_results/distribution_IS_ddim_20000_0.5_objs.npy"
names_path = f"./GenMIP/agents/diffusion_results/distribution_IS_ddim_20000_0.5_names.npy"
# picklex.dump(arr, open(path, 'wb'))

sols = pickle.load(open(sols_path,'rb'))
objs = pickle.load(open(objs_path,'rb'))
names = pickle.load(open(names_path, 'rb'))
names = np.array([name * 50 for name in names])

objs = objs.reshape(-1)
sols = sols.reshape(-1, 1500)
names = names.reshape(-1, 1500)

sols = sols[~np.isnan(objs)][0:1000]
names = names[~np.isnan(objs)][0:1000]
objs = objs[~np.isnan(objs)][0:1000]


pred_x = sols

filename = "./instances/4_independent_set/test/independent_set_0"
coverage = 0.2
obj_vals = []
num_vars = int(pred_x.shape[1] * coverage)
for j in range(pred_x.shape[0]):
    sol_val = {}
    for l in range(pred_x.shape[1]):
        sol_val[names[j,l]] = pred_x[j,l]
    scip_model = Model()
    # add the collected partial solutions to scip and optimize
    # scip_model.setParam('limits/time', 3)
    # scip_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)
    # scip_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
    # scip_model.setParam('heuristics/completesol/addallsols', True)
    scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
    scip_model.setParam('limits/time', 3)
    scip_model.setParam('limits/maxorigsol', 1)
    # scip_model.setParam('limits/solutions', 1)
    scip_model.hideOutput(quiet = True)
    log_path = f'./GenMIP/agents/diffusion_partialsol_logs/independent_set_0_{j}.log'
    scip_model.setLogfile(log_path)
    scip_model.readProblem(f'{filename}.mps')

    s = scip_model.createPartialSol()
    selected_vars = random.sample(range(pred_x.shape[1]), num_vars)
    for k, v in enumerate(scip_model.getVars()):
        if k in selected_vars:
            scip_model.setSolVal(s, v, sol_val[v.name])
    scip_model.addSol(s)
    scip_model.optimize()

    result = is_feasible(log_path)

    if result != False:
        obj_vals.append(result)
    else:
        obj_vals.append(np.nan)

problem_objs = np.array(obj_vals)
pickle.dump(problem_objs, open(f'GenMIP/agents/diffusion_results/distribution_IS_complete_objs.npy', 'wb'))
nan_count = np.isnan(problem_objs).sum()
print(f"instance: {0}, feasible ratio: {(sample_num-nan_count)/sample_num}, obj value: {np.nanmean(problem_objs)}")
