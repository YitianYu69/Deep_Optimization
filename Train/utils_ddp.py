import torch
import torch.distributed as dist

import os

def setup_ddp(rank=None, world_size=None):
    if rank is not None and world_size is not None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)


    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{local_rank}'

    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size, device


def get_ddp_meta():
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{local_rank}'

    return local_rank, global_rank, world_size, device

def rank0():
    return (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)

def clean():
    dist.destroy_process_group()



