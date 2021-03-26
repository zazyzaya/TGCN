import os

import torch 
from torch import nn
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

# Provided by torch in possibly next update for the RPC API 
# but for now, we need to add these ourselves
def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

'''
Because there are some remote parameters in the model,
just calling params() will confuse the optimiser. Instead
we create an RRef for each parameter to tell the opt where
to find it
'''
def _param_rrefs(module):
    rrefs = []
    for param in module.parameters():
        rrefs.append(
            rpc.RRef(param)
        )
    
    return rrefs


class Embedder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Embedder, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(x_dim,h_dim),
            nn.ReLU(),
            nn.Linear(h_dim,z_dim)
        )

    def forward(self, x):
        return self.lin(x)


class RNN(nn.Module):
    def __init__(self, ps, x_dim, h_dim, z_dim):
        super(RNN, self).__init__()

        # Handle for remote embedder (will be VGAE in project)
        self.remote_emb_rref = rpc.remote(
            ps, Embedder, args=(x_dim, h_dim, h_dim)
        )

        # Local RNN to use output of embedder
        self.rnn = nn.GRU(h_dim, h_dim)
        self.lin = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid()
        )

    def forward(self, xs):
        jobs = []

        # Remotely run the embedder on all inputs in xs
        for x in xs:
            jobs.append(
                _remote_method_async(Embedder.forward, self.remote_emb_rref, x)
            )
        
        # Then combine output of embedders for RNN
        xs = torch.stack([j.wait() for j in jobs])
        h = self.rnn(xs)[1]
        return self.lin(h)

    # Like the parameters() function but incorporates params from
    # non-local modules like the embedder
    def parameter_rrefs(self):
        params = []
        params.extend(_remote_method(_param_rrefs, self.remote_emb_rref))
        params.extend(_param_rrefs(self.rnn))

        return params


def gen_toy_data():
    # Only sequences of [1,0,0],[0,1,0],[0,0,1] = 1
    toy_X = torch.rand((3,50,3))

    toy_X[0, :25, 0] = 1
    toy_X[0, :25, 1:] = 0

    toy_X[1, :25, 1] = 1
    toy_X[1, :25, 0] = 0
    toy_X[1, :25, 2] = 0 

    toy_X[2, :25, 2] = 1
    toy_X[2, :25, :2] = 0

    # Add a touch of noise
    noise = torch.rand((3,25,3)) * 0.01
    toy_X[:,:25,:] += noise 

    toy_y = torch.zeros((50,1))
    toy_y[:25] = 1

    return toy_X, toy_y

def training_loop():
    model = RNN('ps', 3, 10, 1)
    X_tr, y_tr = gen_toy_data()
    X_te, y_te = gen_toy_data()

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

        print("[%d] Loss: %0.4f" % (e, loss.item()))

    y_hat = model(X_te)
    y_hat[y_hat < 0.5] = 0
    y_hat[y_hat >= 0.5] = 1

    correct = float((y_hat == y_te).sum().item())
    total = float(y_hat.size(1))
    print("Final accuracy: %d/%d = %0.4f" % (correct, total, correct/total))    


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    if rank == 0:
        rpc.init_rpc('trainer', rank=rank, world_size=world_size)
        training_loop()
    else:
        rpc.init_rpc('ps', rank=rank, world_size=world_size)
        # Do nothing. Get instructions from trainer proc

    # Block until all procs complete
    rpc.shutdown()


if __name__ == '__main__':
    world_size = 2

    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )