from statistics import mean

from torch.distributions import Binomial
from pyscipopt import Model
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


import torch_geometric
from model.cmsp import CMSP
from model.decoder import MipConditionalDecorder
from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
import argparse

from feature import ObservationFunction
from environments import RootPrimalSearch as Environment 
import json
import pickle

from dataset import BipartiteNodeDataWithoutSolution
from utils import extract_bigraph_from_mps

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
    instance = "SC"

    test_size = 100
    sample_num = 10
    output_type = 'save'
    ## sampling parameters

    sampler_type = "ddpm"
    gradient_scale = 100000
    coef = 0.1

    parser = argparse.ArgumentParser()

    parser.add_argument('--instance', type=str, default='SC', help='The instance for testing MIP, SC or CA, CF or IS')
    parser.add_argument('--test_size', type=int, default=100, help='The number of instances')
    parser.add_argument('--sample_nums', type=int, default=30, help='The number of instances')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding size in gnn')
    parser.add_argument('--gcn_layer_num', type=int, default=2, help='The number of layer in gnn')

    parser.add_argument('--time_limit', type=int, default=5, help='The time of scip solving')
    parser.add_argument('--output_type', type=str, default='save', help='print or save')

    args = parser.parse_args()


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
        padding_len = 2000

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
        padding_len = 2000
        
    test_files = []
    for i in range(args.test_size):
        test_files.append(f'./instances/{instance_file}/test/{instance_file[2:]}_{start + i}')
        # test_files.append(f'./new_instances/{instance_file}_4000/{instance_file[2:]}_{i}')

    cmsp_path = f'./model_hub/cmsp{instance_file[1:]}_1.pth'
    cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmps_n_heads, n_layers=cmps_n_layers, padding_len = padding_len, position_emb = position_emb).to(device)
    cmsp.load_state_dict(torch.load(cmsp_path, map_location= device))

    decoder_path = f'./model_hub/decoder{instance_file[1:]}_penalty_1.pth'
    decoder = MipConditionalDecorder(attn_dim=emb_dim,n_heads = decoder_n_heads, n_layers= decoder_n_layers).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location= device))

    ddpm_path = f'./model_hub/ddpm_model{instance_file[1:]}_penalty_1.pth'
    ddpm_model = DDPMTrainer(attn_dim=emb_dim, n_heads= ddpm_n_heads, n_layers= ddpm_n_layers, device=device,
                            timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                            parameterization=ddpm_parameterization)
    ddpm_model.load_state_dict(torch.load(ddpm_path, map_location= device))
    sampler_model = DDIMSampler(trainer_model=ddpm_model, decoder = decoder, gradient_scale=gradient_scale, obj_guided_coef = coef, device=device)

    cmsp.eval()
    decoder.eval()
    ddpm_model.eval()


    observation_function = ObservationFunction()
    env = Environment(observation_function=observation_function, presolve=True)

    
    # feasible_ratios = []
    # mean_obj_vals = []
    obj_values = []
    for i, filename in enumerate(test_files):
        bigraph = extract_bigraph_from_mps(filename, instance_file_type, observation_function, env)
        bigraph = bigraph.to(device)
        A = torch.sparse_coo_tensor(
                bigraph.edge_index,
                bigraph.edge_attr.squeeze(),
                size=(bigraph.constraint_features.shape[0], bigraph.variable_features.shape[0]))
        A = A.index_select(1, bigraph.int_indices)

        b = bigraph.constraint_features[:,0]

        c = bigraph.variable_features[:,0]

        
        obj_vals = []

        # scip
        initial_bound = json.load(open(f'{filename}.json', 'rb'))
        observation_function = ObservationFunction()
        env = Environment(time_limit=args.time_limit, observation_function=observation_function)
        observation, action_set, reward, done, info = env.reset(filename + '.' + instance_file_type, \
                                                                objective_limit=initial_bound['primal_bound'])
        action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(device)
        m = env.model.as_pyscipopt()

        # feasible_num = 0
        # obj_values = []
        problem_obj = []
        for n in range(args.sample_nums):
            mip_features, _, key_padding_mask = cmsp.get_features(bigraph)
            mip_features = mip_features.repeat((sample_num,1,1))
            key_padding_mask = key_padding_mask.repeat((sample_num,1))
            model = env.model.as_pyscipopt()
            torch.cuda.empty_cache()
            pred_x_features, _ = sampler_model.constraint_guided_sample(mip_features, key_padding_mask, A, b, c, S = ddim_timesteps)

            with torch.no_grad():
                pred_x, _ = decoder(mip_features, pred_x_features, key_padding_mask)
            pred_x = torch.round(pred_x).view(sample_num,-1,1)

            penalty = torch.max((A @ pred_x).squeeze() - b, 
                                torch.tensor(0)).sum(axis = 1)
            pred_x.squeeze_()
            
            scip_model = Model()
            # add the collected partial solutions to scip and optimize
            scip_model.setParam('limits/time', args.time_limit)
            scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
            # scip_model.setParam('heuristics/completesol/minimprove', 0.0001)
            scip_model.setObjlimit(initial_bound['primal_bound'])
            # scip_model.setParam('limits/maxorigsol', 1)
            scip_model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
            # scip_model.setParam('heuristics/completesol/solutions', 5)
            # scip_model.setParam('heuristics/completesol/addallsols', True)
            # scip_model.setParam('limits/solutions', 1)

            log_path = f'./GenMIP/agents/new_gnn_logs/{instance_file}_4000/{args.instance}_coverage{args.coverage}_instance{i}_num{n}.log'
            scip_model.setLogfile(log_path)
            scip_model.readProblem(filename + '.' + instance_file_type)

            name_var_dict = {}
            vars = m.getVars(transformed=True)

            for v in scip_model.getVars(transformed=True):
                name_var_dict[v.name] = v
            
            s = scip_model.createPartialSol()
            for j in range(len(pred_x)):
                scip_model.setSolVal(s, vars[i], pred_x[i])
            scip_model.addSol(s)
             
            scip_model.hideOutput(quiet=False)
            scip_model.optimize()

            result = is_feasible(log_path)

            if result != False:
                problem_obj.append(result)
            else:
                problem_obj.append(np.nan)
        problem_obj = np.array(problem_obj)
        nan_count = np.isnan(problem_obj).sum()
        print(f"instance {i}, feasible ratio: {(args.sample_nums-nan_count)/args.sample_nums}, obj value: {np.nanmean(problem_obj)}")
        obj_values.append(problem_obj)
    
    obj_values = np.array(obj_values)
    total_num = args.sample_nums * args.test_size
    nan_count = np.isnan(obj_values).sum()
    print(f'instance: {args.instance}')
    print(f"mean feasible ratio: {(total_num - nan_count)/total_num}, mean obj value: {np.nanmean(obj_values)}")
    
    if args.output_type == 'save':
        pickle.dump(obj_values, open(f'./GenMIP/agents/diffusion_results/{args.instance}_4000_gnn_{args.coverage}.npy', 'wb'))
       









