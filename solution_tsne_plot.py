import torch
import torch_geometric
from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
import numpy as np

from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
from dataset import GraphDataset
from model.decoder import MipConditionalDecorder
from model.cmsp import CMSP


@torch.no_grad()
def ddpm_get_solution(cmsp, decoder, sampler, test_files):
    test_data = GraphDataset(test_files, problem_type=problem_type)
    train_dataloader = torch_geometric.loader.DataLoader(test_data, batch_size=1, shuffle=True)
    # gen_num = 500
    solutions = []
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        x = batch.solution[batch.int_indices]
        mip_features, x_features, key_padding_mask = cmsp.get_features(batch, x)
        pred_emb_solutions, _ = sampler.sample(mip_features, key_padding_mask)
        output, select = decoder(mip_features, pred_emb_solutions, key_padding_mask)
        select = torch.round(output)
        solutions.append(select.cpu().numpy())
        # print(f'Get the number of {j} solution, min value is {torch.min(output)}, max value is {torch.max(output)}')
        print(f'Get the number of {i} solution in {test_files[0]}')
    solutions = np.array(solutions)
    return solutions


def tsne_plot(sols, instance_path, saving_path, tsne):
    sol_embedded = tsne.fit_transform(sols)
    plt.scatter(sol_embedded[:, 0], sol_embedded[:, 1], s=16)
    plt.title(instance_path[2:])
    plt.xlabel("T-SNE1")
    plt.ylabel("T-SNE2")
    plt.savefig(saving_path)
    # plt.show()
    # plt.close()


def get_origin_solution(sol_path):
    sol_file = f'{sol_path}.sol'
    with open(sol_file, "rb") as f:
        sol = pickle.load(f)
    sols = sol['sols']
    return sols


if __name__ == '__main__':
    instance_file = "4_independent_set"
    i = 2
    file_path = f'../samples/{instance_file}/train/{instance_file[2:]}_{i}'
    problem_type = "max"
    sols = get_origin_solution(file_path)
    saving_path = f'agents/plots/tsne_origin_{instance_file[2:]}{i}.jpg'
    tsne = TSNE(n_components=2, learning_rate='auto',
                init='pca', perplexity=30, n_iter=2500)
    tsne.fit(sols)

    # tsne_plot(sols, instance_file, saving_path, tsne)

    # model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # embedding parameters
    emb_num = 3
    emb_dim = 128
    # cmsp parameters
    cmsp_n_heads = 1
    cmsp_n_layers = 1
    padding_len = 1500

    decoder_n_heads = 1
    decoder_n_layers = 2

    is_embeding = True
    ddpm_n_heads = 1
    ddpm_n_layers = 1
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
                                     use_select_net=False).to(device)

    trainer = DDPMTrainer(attn_dim=emb_dim, n_heads=ddpm_n_heads, n_layers=ddpm_n_layers, device=device,
                          timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                          parameterization=ddpm_parameterization)
    trainer.load_state_dict(torch.load(f'../model_hub/ddpm_model_independent_set.pth', map_location=device))
    decoder.load_state_dict(torch.load(f'../model_hub/decoder_independent_set.pth', map_location=device))
    sampler_model = DDIMSampler(trainer_model=trainer, device=device)

    for instance in [0, 1]:
        test_files = []
        for j in range(100):
            test_files.append(f'../samples/{instance_file}/train/{instance_file[2:]}_{instance}.obs')
        sols = ddpm_get_solution(cmsp, decoder, sampler_model, test_files)
        saving_path = f'agents/plots/tsne_ddpm_{instance_file[2:]}_10prob.jpg'
        tsne_plot(sols, instance_file, saving_path, tsne)





