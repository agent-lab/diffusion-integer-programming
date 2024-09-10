import numpy as np
import torch
import torch_sparse
import torch_geometric
from model.cmsp import CMSP
from model.decoder import MipConditionalDecorder
from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
import argparse
import random

from pyscipopt import Model
from feature import ObservationFunction
from environments import RootPrimalSearch as Environment 
import json
import pickle

from dataset import BipartiteNodeDataWithoutSolution
from utils import extract_bigraph_from_mps, is_feasible
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

emb_num = 3
emb_dim = 128

## cmps parameters
cmps_n_heads = 1
cmps_n_layers = 1

position_emb = False

## decoder parameters
decoder_n_heads = 1
decoder_n_layers = 2

## diffusion parameters
is_embeding = True
ddpm_n_heads = 1
ddpm_n_layers = 1
ddpm_timesteps = 1000
ddpm_losstype = "l2"
ddpm_parameterization = "x0"
sampler_loss_type = "l2"
ddim_timesteps = 100

## training parameters
num_epoches = 1000

train_size = 800
valid_size = 100
batch_size = 16
instance = "CF"

test_size = 100
sample_num = 15
output_type = 'save'
## sampling parameters

sampler_type = "ddim"
gradient_scale = 1000
coverage = 0.2
coefs = [0.7]
# obj_guided_coef_ddim = 0.7
# obj_guided_coef_ddpm = 0.3

if instance == 'SC':
    instance_file = '1_set_cover'
    start = 900
    instance_file_type = 'mps'
    problem_type = 'min'
    padding_len = 2000
elif instance == 'CA':
    instance_file = '2_combinatorial_auction'
    start = 900
    instance_file_type = 'mps'
    problem_type = 'max'
    padding_len = 1500

elif instance == 'CF':
    instance_file = '3_capacity_facility'
    start = 900
    instance_file_type = 'mps'
    problem_type = 'min'
    padding_len = 5050

elif instance == 'IS':
    instance_file = '4_independent_set'
    start = 0
    instance_file_type = 'mps'
    problem_type = 'max'
    padding_len = 1500



cmsp_path = f'./model_hub/cmsp{instance_file[1:]}.pth'
cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmps_n_heads, n_layers=cmps_n_layers, padding_len = padding_len, position_emb = position_emb).to(device)
cmsp.load_state_dict(torch.load(cmsp_path, map_location= device))

decoder_path = f'./model_hub/decoder{instance_file[1:]}_penalty.pth'
decoder = MipConditionalDecorder(attn_dim=emb_dim,n_heads = decoder_n_heads, n_layers= decoder_n_layers).to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location= device))

ddpm_path = f'./model_hub/ddpm_model{instance_file[1:]}_penalty.pth'
ddpm_model = DDPMTrainer(attn_dim=emb_dim, n_heads= ddpm_n_heads, n_layers= ddpm_n_layers, device=device,
                          timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                          parameterization=ddpm_parameterization)
ddpm_model.load_state_dict(torch.load(ddpm_path, map_location= device))

test_files = []
for i in range(test_size):
    test_files.append(f'./instances/{instance_file}/test/{instance_file[2:]}_{start+i}')
    #test_files.append(f'./GenMIP/new_instances/{instance_file}/{instance_file[2:]}_{i}')


for coef in coefs:
    if sampler_type == "ddim":
        sampler_model = DDIMSampler(trainer_model=ddpm_model, decoder = decoder, gradient_scale=gradient_scale, obj_guided_coef = coef, device=device)
    else:
        sampler_model = DDPMSampler(trainer_model=ddpm_model, decoder = decoder, gradient_scale=gradient_scale, obj_guided_coef = coef, device=device)

    observation_function = ObservationFunction()
    ## Only NV disable presolve!!!!!!!!
    env = Environment(observation_function=observation_function, presolve=True)


    cmsp.eval()
    decoder.eval()
    ddpm_model.eval()

    cmsp.eval()
    decoder.eval()
    ddpm_model.eval()

    obj_values = []
    times = []
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

        b = bigraph.constraint_features[:,0]

        c = bigraph.variable_features[:,0]

        mip_features, _, key_padding_mask = cmsp.get_features(bigraph)
        mip_features = mip_features.repeat((sample_num,1,1))
        key_padding_mask = key_padding_mask.repeat((sample_num,1))
        model = env.model.as_pyscipopt()
        torch.cuda.empty_cache()
        mid0 = time.time()
        if sampler_type == "ddim":
            pred_x_features, _ = sampler_model.constraint_guided_sample(mip_features, key_padding_mask, A, b, c, S = ddim_timesteps)
        else:
            pred_x_features = sampler_model.constraint_guided_sample(mip_features, key_padding_mask, A, b, c)
        with torch.no_grad():
            pred_x, _ = decoder(mip_features, pred_x_features, key_padding_mask)
        pred_x = torch.round(pred_x).view(sample_num,-1,1)
        penalty = torch.max(torch_sparse.spmm(A.coalesce().indices(), A.coalesce().values(), A.size()[0], A.size()[1], pred_x).squeeze() - b, 
                            torch.tensor(0)).sum(axis = -1)
        pred_x.squeeze_(-1)
        mid1 = time.time()

        obj_vals = []
        num_vars = int(pred_x.shape[1] * coverage)
        for j in range(pred_x.shape[0]):
            sol_val = {}
            for k, v in enumerate(model.getVars(transformed = True)):
                sol_val[v.name[2:]] = int(pred_x[0,k])
            
            

            scip_model = Model()
            # add the collected partial solutions to scip and optimize
            # scip_model.setParam('limits/time', 3)
            # scip_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.AGGRESSIVE)
            # scip_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)
            # scip_model.setParam('heuristics/completesol/addallsols', True)
            scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
            scip_model.setParam('limits/time', 1)
            scip_model.setParam('limits/maxorigsol', 1)
            # scip_model.setParam('limits/solutions', 1)
            scip_model.hideOutput(quiet = True)
            log_path = f'GenMIP/agents/time_logs/{sampler_type}/{instance_file}/{instance}_{sampler_type}_{coverage}_instance{i}_num{j+15}.log'
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

        end = time.time()
        times.append([mid0-start, mid1-mid0, end-mid1])
        problem_objs = np.array(obj_vals)
        print(f"Instance: {i}: objective values: {obj_vals}")
        nan_count = np.isnan(problem_objs).sum()
        print(f"instance: {i}, feasible ratio: {(sample_num-nan_count)/sample_num}, obj value: {np.nanmean(problem_objs)}")
        print(f"total_time {end-start}, average time: {(end-start)/30}, average sample time: {(mid1-mid0)/30}")
        obj_values.append(problem_objs)
        
    time_values = np.array(times)
    obj_values = np.array(obj_values)
    nan_count = np.isnan(obj_values).sum()
    total_num = sample_num * test_size
    print(f"sampler method: {sampler_type}, gradient scale: {gradient_scale}, obj_guided_coef: {coef}")
    print(f"mean feasible ratio: {(total_num - nan_count)/total_num}, mean obj value: {np.nanmean(obj_values)}")
    if output_type == "save":
        pickle.dump(obj_values, open(f'GenMIP/agents/time_results/{sampler_type}/{instance_file}/{instance}_{sampler_type}_{coverage}_obj_1.pkl', 'wb'))
        pickle.dump(time_values, open(f'GenMIP/agents/time_results/{sampler_type}/{instance_file}/{instance}_{sampler_type}_{coverage}_time_1.pkl', 'wb'))