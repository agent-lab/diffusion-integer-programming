import torch
import numpy as np
import json
import argparse

import torch_geometric
from pyscipopt import Model
from torch.distributions.binomial import Binomial
from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
from environments import RootPrimalSearch as Environment
from feature import ObservationFunction
import random
from pyscipopt import SCIP_PARAMSETTING
from model.decoder import MipConditionalDecorder
from model.cmsp import CMSP
from dataset import GraphDataset
from score_matching.model.gnn_model import NeuralDiving



def is_feasible(log_path):
    log_file = open(log_path, "r", encoding='UTF-8')
    lines = log_file.readlines()  # 读取文件的所有行
    assert len(lines) >= 0, '文件行数不足'
    second_line = lines[1].split()  # 获取第二行内容
    if "failed" in second_line:  # 判断第二行是否包含"fail"单词
        return False
    else:
        return True




if __name__ == '__main__':
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    feasible_num = 0
    solution_num = 0
    print("Using {} device".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ddpm', help='The type of the trained model')
    parser.add_argument('--sampler-type', type=str, default='ddim')
    parser.add_argument('--var_num', type=int, default=1500,help='The number of vars in solution')
    parser.add_argument('--var_dim', type=float, default=128, help="1 or 128")
    parser.add_argument('--mip_dim', type=float, default=128, help="")
    parser.add_argument('--ddpm_timesteps', type=float, default=1000, help="")
    parser.add_argument('--ddpm_losstype', type=str, default="l2", help="")
    parser.add_argument('--ddpm_parameterization', type=str, default="x0", help="x0 or eps")
    parser.add_argument('--ddim_steps', type=int, default="100", help="x0 or eps")

    parser.add_argument('--time_limit', type=int, default=1, help='The limit time for solving MIP')
    parser.add_argument('--instance', type=str, default='IS', help='The instance for testing MIP')
    parser.add_argument('--num_instances', type=int, default=100, help='The number of instances')
    parser.add_argument('--num_partial_sols', type=int, default=100, help='The number of partial solutions')
    parser.add_argument('--embeding_size', type=int, default=64, help='')
    parser.add_argument('--gnn_fea_size', type=int, default=64, help='')
    parser.add_argument('--latent_size', type=int, default=64, help='')
    parser.add_argument('--partial_method', type=str, default='selectiveNet', help='')
    parser.add_argument('--coverage', type=float, default=0.2, help='')
    args = parser.parse_args()


    # choose problem instance
    if args.instance == 'IP':
        instance_file = '1_item_placement'
        start = 800
        instance_file_type = '.mps.gz'
    elif args.instance == 'LB':
        instance_file = '2_load_balancing'
        start = 800
        instance_file_type = '.mps.gz'
    elif args.instance == 'AN':
        instance_file = '3_anonymous'
        start = 98
        instance_file_type = '.mps.gz'
    elif args.instance == 'IS':
        instance_file = '4_independent_set'
        start = 0
        instance_file_type = '.mps'
        problem_type = 'max'
    elif args.instance == 'NV':
        instance_file = '5_nn_verification'
        start = 800
        instance_file_type = '.mps'

    test_size = 100
    test_files = []
    batch_size = 1
    for i in range(test_size):
        test_files.append(f'../samples/{instance_file}/test/{instance_file[2:]}_{start + i}.obs')
    test_data = GraphDataset(test_files, problem_type=problem_type)
    test_dataloader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # choose model
    if args.model_type == 'ddpm':
        # embedding parameters
        emb_num = 3
        emb_dim = 128

        # cmsp parameters
        cmsp_n_heads = 1
        cmsp_n_layers = 1
        padding_len = 2000

        decoder_n_heads = 1
        decoder_n_layers = 2

        # diffusion parameters
        is_embeding = True
        ddpm_n_heads = 1
        ddpm_n_layers = 2
        ddpm_timesteps = 1000
        ddpm_losstype = "l2"
        ddpm_parameterization = "x0"
        sampler_type = "ddim"
        sampler_loss_type = "l2"
        ddim_steps = 100

        cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmsp_n_heads, n_layers=cmsp_n_layers,
                    padding_len=padding_len).to(device)
        cmsp.load_state_dict(torch.load(f"../model_hub/cmsp{instance_file[1:]}.pth", map_location=device))
        decoder = MipConditionalDecorder(attn_dim=emb_dim, n_heads=decoder_n_heads, n_layers=decoder_n_layers,
                                         use_select_net=True).to(device)

        trainer = DDPMTrainer(attn_dim=emb_dim, n_heads=ddpm_n_heads, n_layers=ddpm_n_layers, device=device,
                                 timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                                 parameterization=ddpm_parameterization)
        trainer.load_state_dict(torch.load(f'agents/model_hub/ddpm_independent_set.pth', map_location=device))
        decoder.load_state_dict(torch.load(f'agents/model_hub/decoder_independent_set.pth', map_location=device))
        if args.sampler_type == "ddim":
            sampler_model = DDIMSampler(trainer_model=trainer, device=device)
        else:
            sampler_model = DDPMSampler(trainer_model=trainer, device=device)

    elif args.model_type == 'gnn':
        model_file = f'agents/GnnModel/useSelectiveNet-True_C20.0_IS_Neural Diving_model_19.pkl'
        model = NeuralDiving(emb_size=64, gcn_mlp_layer_num=2).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))

    # gaps = np.zeros((args.num_instances, 10*args.time_limit+1))
    bounds = np.zeros((args.num_instances, 10 * args.time_limit + 1))

    num_epochs = 30
    for epoch in range(num_epochs):
        for i, batch in enumerate(test_dataloader):
            solution_num += 1
            instance_file_name = f'../instances/{instance_file}/test/{instance_file[2:]}_{start + i}'
            # create environment, get observation and milp info
            print(f"Testing {instance_file_name}......")
            initial_bound = json.load(open(f'{instance_file_name}.json', 'rb'))
            scip_model = Model()
            observation_function = ObservationFunction()
            env = Environment(time_limit=args.time_limit, observation_function=observation_function)

            if args.model_type != 'scip':
                observation, action_set, reward, done, info = env.reset(instance_file_name + instance_file_type, \
                                                                        objective_limit=initial_bound['primal_bound'])

                actions = []
                num_lps = info['nlps']
                num_sols = 0
                m = env.model.as_pyscipopt()

                # model evalution and sample action (partial solution)
                batch_size = 1
                with torch.no_grad():
                    if args.model_type == 'ddpm':
                        batch = batch.to(device)
                        x = batch.solution[batch.int_indices]
                        with torch.no_grad():
                            mip_features, x_features, key_padding_mask = cmsp.get_features(batch, x)
                        if args.sampler_type == "ddim":
                            pred_emb_solutions, intermediates = sampler_model.sample(mip_features, key_padding_mask)
                        else:
                            pred_emb_solutions = sampler_model.sample(mip_features)

                        output, select = decoder(mip_features, pred_emb_solutions, key_padding_mask)

                        # output = torch.round(output)
                    elif args.model_type == 'gnn':
                        model.eval()
                        output, select = model(
                            batch.constraint_features,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.variable_features
                        )
                action_set = torch.from_numpy(action_set.astype(int))
                output = output[action_set]
                if args.partial_method == 'selectiveNet':
                    select = select[action_set]
                    p = Binomial(1, select)
                    probs = p.sample() * output
                    p = Binomial(1, output[probs > 0])
                    action = (action_set[probs > 0].cpu(), p.sample().cpu())
                else:
                    topK = torch.topk(torch.max(1 - output, output), int(output.shape[0] * args.coverage))[1]
                    p = Binomial(1, output[topK])
                    action = (action_set[topK].cpu(), p.sample().cpu())

                observation, action_set, reward, done, info = env.step(action)

                    # if info['nlps'] > num_lps:
                    #     actions.append(action)
                    #     num_sols += 1
                    # num_lps = info['nlps']

                    # if num_sols == args.num_partial_sols:
                actions.append(action)
                time = info['solvingtime']

                print(f"info: {info['solvingtime']} {num_sols} {info['nlps']}")
                assert actions != []

                # add the collected partial solutions to scip and optimize
                scip_model.setParam('limits/time', args.time_limit)
                scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
                scip_model.setObjlimit(initial_bound['primal_bound'])
                # scip_model.setParam('estimation/restarts/restartpolicy', 'n')
                scip_model.setParam('limits/maxorigsol', args.num_partial_sols)
                scip_model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
                log_path = f'agents/solving_logs/{args.instance}_{args.model_type}_{args.partial_method}{args.coverage}_{i}.log'
                scip_model.setLogfile(log_path)
                scip_model.readProblem(instance_file_name + instance_file_type)

                name_var_dict = {}
                vars = m.getVars(transformed=True)

                for v in scip_model.getVars(transformed=True):
                    name_var_dict[v.name] = v

                for a in actions:
                    s = scip_model.createPartialSol()
                    for j in range(len(a[0])):
                        scip_model.setSolVal(s, name_var_dict[vars[a[0][j]].name[2:]], a[1][j])
                    scip_model.addSol(s)

                scip_model.hideOutput(quiet=True)
                scip_model.optimize()

                if is_feasible(log_path):
                    feasible_num += 1
                print(f'The feasible number is {feasible_num}, total number is{solution_num}, '
                      f'the ratio is {feasible_num / solution_num}')

            else:

                scip_model.setParam('limits/time', args.time_limit)
                scip_model.setObjlimit(initial_bound['primal_bound'])
                scip_model.hideOutput(quiet=True)
                scip_model.setParam('limits/maxsol', 1000)
                scip_model.setParam('limits/maxorigsol', args.num_partial_sols)
                # scip_model.setParam('estimation/restarts/restartpolicy', 'n')
                scip_model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
                scip_model.setLogfile(f'solving_logs/{args.instance}_{args.model_type}_sel.log')
                scip_model.readProblem(instance_file_name + instance_file_type)
                scip_model.optimize()

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




