from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
import torch_geometric
import argparse
from model.diffusion import DDPMSampler, DDIMSampler, DDPMTrainer
from model.decoder import MipConditionalDecorder
from model.cmsp import CMSP
from dataset import GraphDataset

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # embedding parameters
    emb_num = 3
    emb_dim = 128

    # cmsp parameters
    cmsp_n_heads = 1
    cmsp_n_layers = 1
    padding_len = 2000

    ## decoder parameters
    decoder_n_heads = 1
    decoder_n_layers = 2

    ## diffusion parameters
    is_embeding = True
    ddpm_n_heads = 1
    ddpm_n_layers = 2
    ddpm_timesteps = 1000
    ddpm_losstype = "l2"
    ddpm_parameterization = "x0"
    sampler_loss_type = "l2"
    ddim_steps = 100

    ## training parameters
    num_epoches = 1000

    train_size = 100
    batch_size = 1
    instance_path = '4_independent_set'
    start = 0
    instance_file_type = '.mps'
    problem_type = 'max'
    sampler_type = 'ddim'
    model_type = 'ddpm'
    plot_features_size = 8

    train_files = []
    for i in range(train_size):
        train_files.append(f'../samples/{instance_path}/train/{instance_path[2:]}_{i}.obs')
    train_data = GraphDataset(train_files, problem_type = problem_type)
    train_dataloader = torch_geometric.loader.DataLoader(train_data, batch_size=1, shuffle = True)

    cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmsp_n_heads, n_layers=cmsp_n_layers,
                padding_len=padding_len).to(device)
    cmsp.load_state_dict(torch.load(f"../model_hub/cmsp{instance_path[1:]}.pth", map_location=device))
    decoder = MipConditionalDecorder(attn_dim=emb_dim, n_heads=decoder_n_heads, n_layers=decoder_n_layers,
                                        use_select_net=True).to(device)

    trainer = DDPMTrainer(attn_dim=emb_dim, n_heads=ddpm_n_heads, n_layers=ddpm_n_layers, device=device,
                                timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                                parameterization=ddpm_parameterization)
    trainer.load_state_dict(torch.load(f'agents/model_hub/ddpm_independent_set.pth', map_location=device))
    decoder.load_state_dict(torch.load(f'agents/model_hub/decoder_independent_set.pth', map_location=device))
    if sampler_type == "ddim":
        sampler_model = DDIMSampler(trainer_model=trainer, device=device)
    else:
        sampler_model = DDPMSampler(trainer_model=trainer, device=device)



    features = {k:[] for k in range(plot_features_size)}

    for i, batch in enumerate(train_dataloader):
        instance_file_name = f'../instances/{instance_path}/test/{instance_path[2:]}_{start + i}'
        # create environment, get observation and milp info
        print(f"Testing {instance_file_name}......")
        with torch.no_grad():
            batch = batch.to(device)
            x = batch.solution[batch.int_indices]
            mip_features, x_features, key_padding_mask = cmsp.get_features(batch, x)
            if sampler_type == "ddim":
                pred_emb_solutions, intermediates = sampler_model.sample(mip_features, key_padding_mask)
            else:
                pred_emb_solutions = sampler_model.sample(mip_features)

            y = torch.concat([mip_features, pred_emb_solutions], dim=1)
            concat_key_padding_mask = torch.concat([key_padding_mask, key_padding_mask], dim=1)
            for module in decoder.resblocks:
                y = module(y, concat_key_padding_mask)
            y = y[:, -mip_features.shape[1]:, :].squeeze(0)
            mean_y = torch.mean(y, dim=0, keepdim=True).squeeze(0)
            for k in range(plot_features_size):
                features[k].append(mean_y[k])



    fig, axs = plt.subplots(plot_features_size, plot_features_size, figsize=(20, 20), constrained_layout=True)
    # 根据特征画图
    for i in range(plot_features_size):
        for j in range(plot_features_size):
            ax = axs[i, j]
            if i != j:
                ax.scatter(features[i], features[j], s=16)
            else:
                ax.hist(features[j])
            if i == 0 and j == 0:
                ax.legend()
            if i == 0:
                ax.axes.xaxis.set_label_position('top')
                ax.set_xlabel(j)
            if j == 0:
                ax.set_ylabel(i)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
    plt.suptitle(instance_path[2:], fontsize=20)
    plt.savefig(f'agents/plots/ddpm_emb.jpg')
    plt.show()


