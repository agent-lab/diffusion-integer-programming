from model.cmsp import CMSP
import torch
import torch_geometric

from dataset import GraphDataset
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# embedding parameters
emb_num = 3
emb_dim = 128


# cmsp parameters
cmsp_n_heads = 1
cmsp_n_layers = 1

# dataset
train_size = 800
valid_size = 100
batch_size = 64

instance = 'CA'
if instance == 'SC':
    instance_file = '1_set_cover'
    start = 800
    instance_file_type = '.mps'
    problem_type = 'min'
    padding_len = 2000
elif instance == 'CA':
    instance_file = '2_combinatorial_auction'
    start = 800
    instance_file_type = '.mps'
    problem_type = 'max'
    padding_len = 1500

elif instance == 'CF':
    instance_file = '3_capacity_facility'
    start = 800
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
train_dataloader = torch_geometric.loader.DataLoader(train_data, batch_size = batch_size, shuffle= True)

num_epochs = 100
cmsp = CMSP(emb_num=emb_num, emb_dim=emb_dim, n_heads=cmsp_n_heads, n_layers=cmsp_n_layers, padding_len= padding_len).to(device)

optimizer = torch.optim.AdamW([{'params': cmsp.parameters()}], lr = 0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.9)

cross_entropy = torch.nn.CrossEntropyLoss()

cmsp.train()
# cmsp.mip_prenorm(train_dataloader)
total_time = 0
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        t1 = time.time()
        batch = batch.to(device)
        x = batch.solution[batch.int_indices]

        logits_per_mip, logits_per_x, _ = cmsp(batch, x)
        
        contrastive_label = torch.arange(len(batch), device=device)

        mip_loss = cross_entropy(logits_per_mip, contrastive_label)
        x_loss = cross_entropy(logits_per_x, contrastive_label)
        loss = (mip_loss+x_loss)/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        total_time += (t2-t1)
        if (i+1) % 5 == 0:
            print(f"Epoch {epoch}, Iteration {i+1}: loss:{loss}, time: {total_time}")
    scheduler.step()
print(f"total time: {total_time}")
torch.save(cmsp.state_dict(), f"./new_model_hub/cmsp{instance_file[1:]}_batchsize{batch_size}.pth")