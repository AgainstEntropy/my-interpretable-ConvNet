import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    print(f"torch version: {torch.__version__}")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        GPU_nums = torch.cuda.device_count()
        GPU = torch.cuda.get_device_properties(0)
        print(f"There are {GPU_nums} GPUs in total.\nThe first GPU is: {GPU}")
        if '3060' in GPU.name:
            print(f"CUDA version: {torch.cuda_version}")
        else:
            print(f"CUDA version: {torch.version.cuda}")
    device = torch.device(f"cuda:{0}" if use_cuda else "cpu")
    print(f"Using {device} now!")
