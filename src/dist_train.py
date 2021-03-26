import os

import torch 
from torch import nn
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

from models.distributed.tgcn import TGCN, get_remote_gae

def init_procs(rank, world_size, x_dim, h_dim, tr_args={}):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:42068'

    # Master (RNN module)
    if rank == world_size-1:
        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        rrefs = []
        for i in range(world_size-1):
            rrefs.append(
                rpc.remote(
                    'worker'+str(i),
                    get_remote_gae,
                    args=(x_dim, h_dim)
                )
            )

        train(rrefs, **tr_args)

    # Slaves
    else:
        # Slaves are their own process group. This allows
        # DDP to work between these processes
        dist.init_process_group(
            'gloo', rank=rank, 
            world_size=world_size-1
        )

        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    # Block until all procs complete
    rpc.shutdown()


def train(rrefs, **kwargs):
    pass