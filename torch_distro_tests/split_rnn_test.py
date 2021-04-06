import os
import asyncio
import time 
from random import random 

import torch 
from torch import device, nn
import torch.distributed.rpc as rpc 
import torch.distributed as dist 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.optim import Adam

from test_rpc import _remote_method_async, _remote_method, _param_rrefs, gen_toy_data

class Embedder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Embedder, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(x_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,z_dim)
        )

    def forward(self, x):
        time.sleep(random()*0.01)
        print(rpc.get_worker_info().name + ' embedding')
        return self.lin(x)


def init_embedder(x_dim, h_dim):
    model = Embedder(x_dim, h_dim, h_dim)
    return DDP(model)

class RNNMulti(nn.Module):
    def __init__(self, rrefs, h_dim, z_dim):
        super(RNNMulti, self).__init__()
        
        # Do the embedding on several DDP embedders hosted on
        # external machines
        self.remote_embs = rrefs
        self.n_workers = len(rrefs)
        
        # Do the sequential RNN portion on 1 machine
        self.rnn = nn.GRU(h_dim, h_dim)
        self.out = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid()
        )
        
    def forward(self, xs):
        x_futs = []
        for i in range(xs.size(0)):
            x_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.remote_embs[i % self.n_workers],
                    xs[i]
                )
            )

        xs = torch.stack([f.wait() for f in x_futs])
        print(rpc.get_worker_info().name + ' running RNN')
        h = self.rnn(xs)[1]
        return self.out(h).squeeze(0)


    def parameter_rrefs(self):
        params = []
        for remote in self.remote_embs:
            params.extend(_remote_method(_param_rrefs, remote))

        params.extend(_param_rrefs(self.rnn))
        params.extend(_param_rrefs(self.out))
        return params


def train_loop(rrefs, X_tr, y_tr, X_te, y_te):
    model = RNNMulti(rrefs, 10, 1)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=0.01
    )

    loss_fn = nn.MSELoss()

    for e in range(100):
        with dist_autograd.context() as context_id:
            y_hat = model(X_tr)
            loss = loss_fn(y_hat, y_tr)

            dist_autograd.backward(context_id, [loss])
            opt.step(context_id)
            # No need to zero grad because it's blown 
            # away every step by the dist API 

        print("[%d] Loss: %0.4f\n" % (e, loss.item()))

    y_hat = model(X_te)
    y_hat[y_hat < 0.5] = 0
    y_hat[y_hat >= 0.5] = 1

    correct = float((y_hat == y_te).sum().item())
    total = float(y_hat.size(0))
    print("Final accuracy: %d/%d = %0.4f" % (correct, total, correct/total)) 


def run_worker(rank, world_size):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:42068'

    # Master
    if rank == world_size-1:
        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        X_tr, y_tr = gen_toy_data()
        X_te, y_te = gen_toy_data()

        rrefs = []
        for i in range(world_size-1):
            rrefs.append(
                rpc.remote(
                    'worker'+str(i),
                    init_embedder,
                    args=(X_tr.size(2), 10)
                )
            )

        train_loop(rrefs, X_tr, y_tr, X_te, y_te)

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


if __name__ == '__main__':
    world_size = 4

    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
