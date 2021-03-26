import datetime
import os 
import random 
from time import sleep

import torch 
from torch import nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import irecv
from torch.multiprocessing import Process 

TCP_ADDR = 'localhost'
TCP_PORT = '42069'
TIMEOUT = 30

def init_process(rank, world_size, fn, backend='gloo'):
    print("%d Started process rank %d" % (os.getpid(), rank))

    dist.init_process_group(
        backend, 
        init_method="tcp://%s:%s" % (TCP_ADDR, TCP_PORT),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=TIMEOUT)
    )

    fn(rank, world_size)

def start_processes(world_size, fn, **kwargs):
    procs = []
    
    for r in range(world_size):
        p = Process(target=init_process, args=(r, world_size, fn))
        p.start()
        procs.append(p)

    for p in procs:
        p.join() 

    print("Donecakes")


def hello_world(rank, ws):
    sl = random.random()
    sleep(sl)
    print("Hello world from %d!" % rank)


# Outputs eye matrix 
def message_pass(rank, ws):
    if rank == 0:
        my_tensors = [torch.zeros(ws, requires_grad=True) for _ in range(ws)]
        
        # Can't pass messages to self, so do any work within proc
        my_tensors[0][0] = 1

        # Wait for messages from other procs
        msgs = [dist.irecv(my_tensors[i], src=i) for i in range(1,ws)]
        [msg.wait() for msg in msgs]

        print(my_tensors)
    else:
        # Send master onehot tensor of rank
        t = torch.zeros(ws, requires_grad=True)
        t[rank] = 1
        t = torch.sigmoid(t)
        dist.isend(t, 0)


def gather(rank, ws):
    t_list = [torch.zeros(ws, requires_grad=True) for _ in range(ws)]
    
    to_send = torch.zeros(ws, requires_grad=True)
    to_send[rank] = 1
    to_send = torch.sigmoid(to_send)
    print(to_send)

    dist.all_gather(t_list, to_send)

    if rank == 0:
        print()
        [print(t) for t in t_list]
        print()

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

class DistRNN(nn.Module):
    def __init__(self, x_dim=3, hidden=3):
        super(DistRNN, self).__init__()

        self.lin = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU()
        )

        self.rnn = nn.GRU(hidden, hidden)
        self.out = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, X, recurrent=False):
        if recurrent:
            return self.out(
                self.rnn(X)[1]
            )
        else:
            return self.lin(X)


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

def test_models():
    pass

start_processes(5,gather)