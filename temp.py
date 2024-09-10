import pickle
import numpy as np
# arr = np.zeros((100,30))
# path1 = f"GenMIP/agents/diffusion_results/SC3000_ddim_150000_0.9_1.npy"
# arr_1 = pickle.load(open(path1,'rb'))
# path2 = f"GenMIP/agents/diffusion_results/SC3000_ddim_150000_0.9_2.npy"
# arr_2 = pickle.load(open(path2,'rb'))
# path3 = f"GenMIP/agents/diffusion_results/SC3000_ddim_150000_0.9_3.npy"
# arr_3 = pickle.load(open(path3,'rb'))
# arr[:,:10] = arr_1 
# arr[:,10:20] = arr_2
# arr[:,20:] = arr_3

# print(arr.shape)
path = f"GenMIP/agents/SC_ddim_100000_0.9_0.2_partial_sols.npy"
# picklex.dump(arr, open(path, 'wb'))

res = pickle.load(open(path,'rb'))
nan_count = np.isnan(res).sum()
print(f"ratio: {1-(nan_count/3000)}")
print(f"value: {np.nanmean(res)}+-{np.nanstd(res)}")