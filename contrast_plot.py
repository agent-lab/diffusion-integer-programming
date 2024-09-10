import matplotlib.pyplot as plt
import pickle

import numpy as np
def get_ratio(arr):
    res = []
    for i in range(100):
        res.append(1-np.isnan(arr[i]).sum()/30)
    return res
def get_data(instance):
    if instance == 'SC':
        instance_file = '1_set_cover'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
        low_gnn_result_path = f"results/orig_gnn_results/SC_gnn_0.2.npy"
        upper_gnn_result_path = f"results/orig_gnn_results/SC_gnn_0.3.npy"
        ddim_result_path = f"results/diffusion_results/SC_ddim_100000_0.9_100.npy"
        ddpm_result_path = f"results/diffusion_results/SC_ddpm_15000_0.1_100.npy"
    elif instance == 'CA':
        instance_file = '2_combinatorial_auction'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'max'
        low_gnn_result_path = f"results/orig_gnn_results/CA_gnn_0.2.npy"
        upper_gnn_result_path = f"results/orig_gnn_results/CA_gnn_0.3.npy"
        ddim_result_path = f"results/diffusion_results/CA_ddim_20000_0.7_100.npy"
        ddpm_result_path = f"results/diffusion_results/CA_ddpm_10000_0.3_100.npy"
    elif instance == 'CF':
        instance_file = '3_capacity_facility'
        start = 900
        instance_file_type = 'mps'
        problem_type = 'min'
        low_gnn_result_path = f"results/new_gnn_results/CF_gnn_0.1.npy"
        upper_gnn_result_path = f"results/orig_gnn_results/CF_gnn_0.2.npy"
        ddim_result_path = f"results/diffusion_results/CF_ddim_1000_0.7_100.npy"
        ddpm_result_path = f"results/diffusion_results/CF_ddpm_500000_0.1_100.npy"
    elif instance == 'IS':
        instance_file = '4_independent_set'
        start = 0
        instance_file_type = 'mps'
        problem_type = 'max'
        low_gnn_result_path = f"results/orig_gnn_results/IS_gnn_0.2.npy"
        upper_gnn_result_path = f"results/orig_gnn_results/IS_gnn_0.3.npy"
        ddim_result_path = f"results/diffusion_results/IS_ddim_20000_0.5_100.npy"
        ddpm_result_path = f"results/diffusion_results/IS_ddpm_10000_0.1_100.npy"
    scip_result_path = f"results/scip_results/{instance}_scip.npy"
    low_gnn_result = pickle.load(open(low_gnn_result_path, 'rb'))
    upper_gnn_result = pickle.load(open(upper_gnn_result_path, 'rb'))
    ddim_result = pickle.load(open(ddim_result_path, 'rb'))
    ddpm_result = pickle.load(open(ddpm_result_path, 'rb'))
    # low_gnn_res = get_ratio(low_gnn_result)
    # upper_gnn_res = get_ratio(upper_gnn_result)
    # ddim_res = get_ratio(ddim_result)
    # ddpm_res = get_ratio(ddpm_result)
    return (~np.isnan(low_gnn_result)), (~np.isnan(upper_gnn_result)), (~np.isnan(ddim_result)), (~np.isnan(ddpm_result))
    
instances = {1: "SC", 2: "CA", 3: "CF", 4: "IS"}

fig, axes = plt.subplots(1, 4, figsize=[16,4],  sharey=True)

# 设置滑动窗口的大小
window_size = 5

def sliding_average(data, window_size):
    pad_size = window_size // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    weights = np.ones(window_size) / window_size
    sliding_avg = np.convolve(padded_data, weights, mode='valid')
    return sliding_avg

for i, ax in enumerate(axes):
    instance = instances[i + 1]

    low_gnn_res, upper_gnn_res, ddim_res, ddpm_res = get_data(instance)
    mean_low_gnn_res = np.mean(low_gnn_res, axis = 1)
    mean_upper_gnn_res = np.mean(upper_gnn_res, axis = 1)
    mean_ddim_res = np.mean(ddim_res, axis = 1)
    mean_ddpm_res = np.mean(ddpm_res, axis = 1)
    x = np.arange(1, 101)
    low_gnn_smooth = sliding_average(mean_low_gnn_res, window_size)
    upper_gnn_smooth = sliding_average(mean_upper_gnn_res, window_size)
    ddim_smooth = sliding_average(mean_ddim_res, window_size)
    ddpm_smooth = sliding_average(mean_ddpm_res, window_size)

    ax.plot(x, low_gnn_smooth, linewidth=2, label="ND (low coverage)")
    ax.plot(x, upper_gnn_smooth, linewidth=2, label="ND (high coverage)")
    ax.plot(x, ddim_smooth, linewidth=2, label="IP Guided DDIM")
    ax.plot(x, ddpm_smooth, linewidth=2, label="IP Guided DDPM")

    # std_low_gnn_res = np.std(low_gnn_res, axis = 1)
    # std_upper_gnn_res = np.std(upper_gnn_res, axis = 1)
    # std_ddim_res = np.std(ddim_res, axis = 1)
    # std_ddpm_res = np.std(ddpm_res, axis = 1)

    # ax.fill_between(x, low_gnn_smooth - std_low_gnn_res, low_gnn_smooth + std_low_gnn_res, alpha=0.2)
    # ax.fill_between(x, upper_gnn_smooth - std_upper_gnn_res, upper_gnn_smooth + std_upper_gnn_res, alpha=0.2)
    # ax.fill_between(x, ddim_smooth - std_ddim_res, ddim_smooth + std_ddim_res, alpha=0.2)
    # ax.fill_between(x, ddpm_smooth - std_ddpm_res, ddpm_smooth + std_ddpm_res, alpha=0.2)

    # ax.set_aspect(100)
    ax.set_xlabel('Instance')
    if i == 0:
        ax.set_ylabel("Average Feasible Ratio")
        # ax.legend(loc="best", prop={'size': 6})
    ax.set_title(instance)
legend = ['ND (low coverage)','ND (high coverage)','IP Guided DDIM', 'IP Guided DDPM']
plt.subplots_adjust( bottom=0.2, left = 0.05, right = 0.95)
fig.legend(legend, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4)
plt.show()
plt.savefig('contrast_plot_new.pdf')