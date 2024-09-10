from dataset import GraphDataset
import torch
import torch_geometric
from model.cmsp import CMSP

from model.decoder import MipConditionalDecorder
from model.diffusion import DDPMTrainer, DDPMSampler, DDIMSampler, get_clip_loss
import argparse

from utils import SelectiveNetLoss

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
use_select_net = False

## diffusion parameters
is_embeding = True
ddpm_n_heads = 1
ddpm_n_layers = 1
ddpm_timesteps = 1000
ddpm_losstype = "l2"
ddpm_parameterization = "x0"
sampler_type = "ddpm"
sampler_loss_type = "l2"
ddim_steps = 100

## training parameters
num_epoches = 100
penalty_coef = 2000
use_select_net = False
use_comtrastive_learning = False


## dataset
train_size = 800
valid_size = 100
batch_size = 8

instance = 'CF'
if instance == 'SC':
    instance_file = '1_set_cover'
    start = 900
    instance_file_type = '.mps'
    problem_type = 'min'
    padding_len = 2000
elif instance == 'CA':
    instance_file = '2_combinatorial_auction'
    start = 900
    instance_file_type = '.mps'
    problem_type = 'max'
    padding_len = 2000

elif instance == 'CF':
    instance_file = '3_capacity_facility'
    start = 900
    instance_file_type = '.mps'
    problem_type = 'min'
    padding_len = 5050

elif instance == 'IS':
    instance_file = '4_independent_set'
    start = 0
    instance_file_type = '.mps'
    problem_type = 'max'
    padding_len = 1500


train_files = []
for i in range(train_size):
    train_files.append(f'./samples/{instance_file}/train/{instance_file[2:]}_{i}.obs')
valid_files = []
for i in range(valid_size):
    valid_files.append(f'./samples/{instance_file}/valid/{instance_file[2:]}_{start+i}.obs')
train_data = GraphDataset(train_files, problem_type = problem_type)
train_dataloader = torch_geometric.loader.DataLoader(train_data, batch_size = batch_size, shuffle = True)


cmsp_path = f'./model_hub/cmsp{instance_file[1:]}_batchsize32.pth'
cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmps_n_heads, n_layers=cmps_n_layers, padding_len = padding_len, position_emb = position_emb).to(device)
if use_comtrastive_learning:
    cmsp.load_state_dict(torch.load(cmsp_path, map_location= device))

# decoder_path = f'./model_hub/decoder{instance_file[1:]}.pth'
decoder = MipConditionalDecorder(attn_dim=emb_dim,n_heads = decoder_n_heads, n_layers= decoder_n_layers, use_select_net=use_select_net).to(device)
# decoder.load_state_dict(torch.load(decoder_path, map_location= device))

bce_loss = torch.nn.BCELoss(reduction= 'none')
mean_aggr = torch_geometric.nn.aggr.MeanAggregation()
select_loss = SelectiveNetLoss()
# decoder.load_state_dict(torch.load(decoder_path, map_location= device))

ddpm_model = DDPMTrainer(attn_dim=emb_dim, n_heads= ddpm_n_heads, n_layers= ddpm_n_layers, device=device,
                          timesteps=ddpm_timesteps, loss_type=ddpm_losstype,
                          parameterization=ddpm_parameterization)
                          
optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer.add_param_group({'params':decoder.parameters(), 'lr':1e-3})
if ~use_comtrastive_learning:
    optimizer.add_param_group({'params':cmsp.parameters(), 'lr':1e-3})
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.9)

if use_comtrastive_learning:
    cmsp.eval()
else:
    cmsp.train()
decoder.train()
ddpm_model.train()
total_time = 0
for epoch in range(num_epoches):
    for i, batch in enumerate(train_dataloader):
        t1 = time.time()
        batch = batch.to(device)
        x = batch.solution[batch.int_indices]

        if use_comtrastive_learning:
            mip_features, x_features, key_padding_mask = cmsp.get_features(batch, x)
        else:
            mip_features, x_features, key_padding_mask = cmsp.get_features_require_grad(batch, x)

        pre_x_features, mse_loss_value = ddpm_model(
            x_features,
            mip_features,
            key_padding_mask
        )
        pred_x, selected_vars = decoder(mip_features, pre_x_features, key_padding_mask)
        if use_select_net:
        # bce = bce_loss(pred_x, x.float())
        # bce_loss_value = mean_aggr(bce.unsqueeze(dim=1), batch.int_vars_batch).mean()
            bce_loss_value = select_loss.compute(pred_x, selected_vars, batch).mean()
        else:
            bce = bce_loss(pred_x, x.float())
            bce_loss_value = mean_aggr(bce.unsqueeze(dim=1), batch.int_vars_batch).mean()
        
        A = torch.sparse_coo_tensor(
            batch.edge_index,
            batch.edge_attr.squeeze(),
            size=(batch.constraint_features.shape[0], batch.variable_features.shape[0]))
        A = A.index_select(1, batch.int_indices)
        penalty = penalty_coef*torch.max((A.mm(pred_x.unsqueeze(dim=1)).squeeze() - batch.constraint_features[:,0]), 
                            torch.tensor(0)).mean()

        loss = mse_loss_value + bce_loss_value + penalty
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        t2 = time.time()
        total_time += (t2-t1)
        if (i + 1) % 5 == 0:
            print(f'Epoch {epoch}, Iteration {i + 1}: training mse loss: {mse_loss_value}, corresponding bce loss: {bce_loss_value}, penalty:{penalty}, total loss: {loss}, total_time:{total_time}')
    scheduler.step()
    
if ~use_comtrastive_learning:
    torch.save(cmsp.state_dict(), f"./new_model_hub/Constrastive_{use_comtrastive_learning}_CMSP_model{instance_file[1:]}.pth")
torch.save(ddpm_model.state_dict(), f"./new_model_hub/Constrastive_{use_comtrastive_learning}_ddpm_model{instance_file[1:]}.pth")
torch.save(decoder.state_dict(), f"./new_model_hub/Constrastive_{use_comtrastive_learning}_decoder{instance_file[1:]}.pth")
print(f"total time: {total_time}")