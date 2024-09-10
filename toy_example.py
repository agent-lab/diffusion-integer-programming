import ecole
import pyscipopt

import numpy as np
import torch
import torch_geometric
from model.cmsp import CMSP
from model.decoder import MipConditionalDecorder
from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
from gnn_model.gnn_model import NeuralDiving
from torch.distributions import Binomial
import argparse

from feature import ObservationFunction
from environments import RootPrimalSearch as Environment 
import json
import pickle
from pyscipopt import Model

from dataset import BipartiteNodeDataWithoutSolution
from utils import extract_bigraph_from_mps
import random

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 34
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed+34)
ecole.seed(seed)
observation_function = ObservationFunction()
env = Environment(observation_function=observation_function, presolve=False, toy_example = True)
bigraph = extract_bigraph_from_mps('toy_example', 'mps', observation_function, env)

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

test_size = 10
sample_num = 10
output_type = 'save'
## sampling parameters

sampler_type = "ddim"

# obj_guided_coef_ddim = 0.7
# obj_guided_coef_ddpm = 0.3

instance_file = '4_independent_set'
start = 0
instance_file_type = 'mps'
problem_type = 'max'
padding_len = 2000

sampler_type = 'ddim'

gradient_scale = 20000
coef = 0.5

cmsp_path = f'../model_hub/cmsp{instance_file[1:]}.pth'
cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmps_n_heads, n_layers=cmps_n_layers, padding_len = padding_len, position_emb = position_emb).to(device)
cmsp.load_state_dict(torch.load(cmsp_path, map_location= device))

decoder_path = f'../model_hub/decoder{instance_file[1:]}.pth'
decoder = MipConditionalDecorder(attn_dim=emb_dim,n_heads = decoder_n_heads, n_layers= decoder_n_layers).to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location= device))

ddpm_path = f'../model_hub/ddpm_model{instance_file[1:]}.pth'
ddpm_model = DDPMTrainer(attn_dim=emb_dim, n_heads= ddpm_n_heads, n_layers= ddpm_n_layers, device=device,
                          timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                          parameterization=ddpm_parameterization)
ddpm_model.load_state_dict(torch.load(ddpm_path, map_location= device))

if sampler_type == "ddim":
    sampler_model = DDIMSampler(trainer_model=ddpm_model, decoder = decoder, gradient_scale=gradient_scale, obj_guided_coef = coef, device=device)
else:
    sampler_model = DDPMSampler(trainer_model=ddpm_model, decoder = decoder, gradient_scale=gradient_scale, obj_guided_coef = coef, device=device)

gnn_path = f'../model_hub/useSelectiveNet-True_C20.0_IS_Neural Diving_model_99.pkl'
gnn_model = NeuralDiving(emb_size=emb_dim, gcn_mlp_layer_num=2).to(device)
gnn_model.load_state_dict(torch.load(gnn_path, map_location= device))


bigraph = bigraph.to(device)
sample_num = 1
A = torch.sparse_coo_tensor(
    bigraph.edge_index,
    bigraph.edge_attr.squeeze(),
    size=(bigraph.constraint_features.shape[0], bigraph.variable_features.shape[0]))

b = bigraph.constraint_features[:,0]

c = bigraph.variable_features[:,0]
mip_features, _, key_padding_mask = cmsp.get_features(bigraph)
mip_features = mip_features.repeat((sample_num,1,1))
key_padding_mask = key_padding_mask.repeat((sample_num,1))
model = env.model.as_pyscipopt()
torch.cuda.empty_cache()

pred_x_features_guided, intermediates_guided = sampler_model.constraint_guided_sample(mip_features, key_padding_mask, A, b, c, S = ddim_timesteps)
pred_x_features, intermediates = sampler_model.sample(mip_features, key_padding_mask)

output, select = gnn_model(
            bigraph.constraint_features,
            bigraph.edge_index,
            bigraph.edge_attr,
            bigraph.variable_features
        )
    
steps = [0, 10, 50, 100]
values_no_guided = np.zeros((4, 15))
for i in range(len(steps)):
    pred_x, _ = decoder(mip_features, intermediates[steps[i]], key_padding_mask)
    j = 0
    for v in model.getVars(transformed = True):
        values_no_guided[i, int(v.name.split("_")[-1])] = pred_x[j]
        j += 1
values_guided = np.zeros((4, 15))
for i in range(len(steps)):
    pred_x, _ = decoder(mip_features, intermediates_guided[steps[i]], key_padding_mask)
    j = 0
    for v in model.getVars(transformed = True):
        values_guided[i, int(v.name.split("_")[-1])] = pred_x[j]
        j += 1

values_gnn = np.zeros((4,15))
j = 0
for v in model.getVars(transformed = True):
    values_gnn[0, int(v.name.split("_")[-1])] = output[j]
    j += 1
p = Binomial(1, output)
sol = p.sample().cpu().detach().numpy()
j = 0
for v in model.getVars(transformed = True):
    values_gnn[1, int(v.name.split("_")[-1])] = sol[j]
    j += 1
action_set = np.arange(15)
# p = Binomial(1, select)
# probs = p.sample() * output
probs = (select > 0.015) * output
probs = probs.cpu()
p = Binomial(1, output[probs > 0])



action = (action_set[probs > 0], p.sample().cpu())
d = {}
j = 0
for v in model.getVars(transformed = True):
    values_gnn[2, int(v.name.split("_")[-1])] = output[j]
    d[j] = int(v.name.split("_")[-1])
    j += 1
z = 0
partial_sol = {}
vars = model.getVars(transformed = True)
for a in action[0]:
    values_gnn[2, d[a]] = action[1][z]
    partial_sol[vars[a].name[2:]] = action[1][z].item()
    z += 1
print(partial_sol, output, select)



scip_model = Model()
scip_model.readProblem('./toy_example.mps')
scip_model.setParam('heuristics/completesol/maxunknownrate', 0.999)
scip_model.setParam('limits/solutions', 2)
name_var_dict = {}
vars = scip_model.getVars(transformed=True)

for v in scip_model.getVars(transformed=True):
    name_var_dict[v.name] = v

s = scip_model.createPartialSol()
z = 0
for a in partial_sol.keys():
    scip_model.setSolVal(s, name_var_dict[a], partial_sol[a])
    z += 1
scip_model.addSol(s)
scip_model.optimize()

s = scip_model.getBestSol()
for v in scip_model.getVars(transformed = True):
    values_gnn[3, int(v.name.split('_')[-1])] = scip_model.getSolVal(s, v)

# 创建一个包含两个子图的图形
fig, axs = plt.subplots(3, 5,  gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.1], 'height_ratios':[1,1,1]}, figsize=(9, 6))

# 定义颜色映射
cmap = cm.get_cmap('coolwarm')  

values = [values_gnn, values_no_guided, values_guided]

title = [
    ['predicted solution', 'random sampling', 'partial solution', 'complete solution'],
    ['step 0', 'step 10', 'step 50', 'step 100'],
    ['step 0', 'step 10', 'step 50', 'step 100']
]

for k in range(len(values)):

    for i in range(len(axs[k])-1):
        G = nx.Graph()

        for j in range(15):
            # if j==9:
            #     continue
            G.add_node(j, value=values[k][i,j])

        G.add_edges_from([(13,1), (13,6), (1,6), (0,10), (3,8), (14,2), (0,13), (5, 7), (0, 5), (0,8), (1,14), (1,7), (1,11), (1,2),
                    (2,13), (3,10), (3,6), (4,8), (6,10), (10,12), (11,13), (13,14)])

        # 获取节点的值
        node_values = nx.get_node_attributes(G, 'value')

        # 可以根据需要选择其他颜色映射

        # 将值映射到颜色
        node_colors = [cmap(value) for value in node_values.values()]


        # 绘制图形
        pos = nx.spring_layout(G, seed=seed)  # 定义节点位置
        pos[9] = (0.2,0.25)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, ax=axs[k, i], node_size=68)
        nx.draw_networkx_edges(G, pos, ax=axs[k,i])
        nx.draw_networkx_labels(G, pos, ax=axs[k,i], font_size=7)
        axs[k, i].set_title(title[k][i], loc = 'center', fontsize = 10)
        

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axs[k, -1])
    cbar.set_label('Variable Value')

    plt.figtext(0.038, 0.82, 'Neural Diving', fontsize=10)
    plt.figtext(0.023, 0.50, 'Unguided DDIM', fontsize=10)
    plt.figtext(0.02, 0.17, 'IP Guided DDIM', fontsize=10)

    plt.subplots_adjust(wspace=0.1, left=0.18, hspace=0.3)

# 显示图形
plt.subplots_adjust(top=0.95, bottom=0.05, left = 0.15, right = 0.9)
plt.show()
plt.savefig(f'toy_example_{seed}.pdf')