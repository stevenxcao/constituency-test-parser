import torch

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using CUDA!")
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy
    
def torch_load(load_path):
    if use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location = 
                          lambda storage, location: storage)
