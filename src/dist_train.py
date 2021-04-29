import os
import pickle
import time

import numpy as np
import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

import generators as g
import loaders.load_lanl_dist as ld 
from distmodel.utils import _remote_method_async, _remote_method
from distmodel.tgcn_pred import TGCN, GAE_DDP, get_remote_gae
from utils import get_score, tpr_fpr, get_optimal_cutoff

DEFAULTS = {
    'h_size': 64, 
    'z_size': 32,
    'lr': 0.001,
    'epochs': 1,
    'min': 25,
    'patience': 5,
    'n_gru': 2,
    'nratio': 10,
    'val_nratio': 1
}

WORKERS=8
W_THREADS=1
M_THREADS=1

DELTA=int((60**2) * 2)
TR_START=0
TR_END=ld.DATE_OF_EVIL_LANL-DELTA*2

VAL_START=TR_END
VAL_END=VAL_START+DELTA*2

TE_START=ld.DATE_OF_EVIL_LANL
#TE_END = 228642 # First 20 anoms
TE_END = 740104 # First 100 anoms
#TE_END = 1089597 # First 500 anoms
#TE_END = 5011200 # Full

TE_DELTA=DELTA

torch.set_num_threads(1)

'''
Constructs params for data loaders
'''
def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = (end-start) // delta
    slices_needed += 1

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    remainder = slices_needed % num_workers 
    if remainder:
        # Make sure worker 0 has at least 2 deltas to ensure it can 
        # run prediction properly (if no remainder, the check on max_workers
        # ensures worker0 has at least 2)
        if per_worker[0] == 1:
            per_worker[0] = 2
            remainder -= 1

        # Then put remaining tasks on last workers since it's likely the 
        # final timeslice is stopped halfway (ie it's less than a delta
        # so giving it extra timesteps is more likely okay)
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1 

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start
    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            kwargs.append({
                # If pred this is the only way to ensure all edges are considered
                # Otherwise the first delta on every worker is ignored (also note:
                # this is why worker 0 must have at least 2 deltas, as it must
                # start at 0)
                'start': max(0, prev-delta),    
                'end': end_t,
                'delta': delta, 
                'is_test': isTe,
                'jobs': min(W_THREADS, 8)
            })
            prev = end_t

    return kwargs
    

def init_workers(num_workers, h_dim, start, end, delta, isTe):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                get_remote_gae,
                args=(h_dim, ld.load_lanl_dist, kwargs[i]),
            )
        )

    return rrefs

def init_procs(rank, world_size, tr_args=DEFAULTS):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:42068'

    # Master (RNN module)
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)

        # Master gets 16 threads and 4x4 threaded workers
        # In theory, only 16 threads should run at a time while
        # master sleeps, waiting on worker procs
        #torch.set_num_threads(16)

        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

        rrefs = init_workers(
            world_size-1, tr_args['h_size'], 
            TR_START, TR_END, DELTA, False
        )

        model, zs, h0 = train(rrefs, tr_args)
        get_cutoff(model, h0, tr_args)
        test(model, zs, h0, rrefs)

    # Slaves
    else:
        # If there are 4 workers, give them each 4 threads 
        # (Total 16 is equal to serial model)
        torch.set_num_threads(W_THREADS)
        
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

'''
Given a trained model, generate the optimal cutoff point using
the validation data
'''
def get_cutoff(model, h0, kwargs):
    # First load validation data onto one of the GCNs
    _remote_method(
        GAE_DDP.load_new_data,
        model.gcns[0],
        ld.load_lanl_dist,
        {
            'start': VAL_START,
            'end': VAL_END,
            'delta': DELTA,
            'jobs': 2,
            'is_test': False
        }
    )

    # Then generate GCN embeds
    model.eval()
    zs = _remote_method(
        GAE_DDP.forward,
        model.gcns[0], 
        ld.LANL_Data.ALL,
        True
    )

    # Finally, generate actual embeds
    with torch.no_grad():
        zs = model.gru(
            torch.tanh(zs),
            h0
        )

    # Then score them
    p,n = _remote_method(
        GAE_DDP.score_edges, 
        model.gcns[0],
        zs, ld.LANL_Data.ALL,
        kwargs['val_nratio']
    )

    # Finally, figure out the optimal cutoff score
    model.cutoff = get_optimal_cutoff(p,n)


def train(rrefs, kwargs):
    model = TGCN(
        rrefs, kwargs['h_size'], kwargs['z_size'], 
        gru_hidden_units=kwargs['n_gru']
    )

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (None, 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        with dist_autograd.context() as context_id:
            st = time.time()
            zs = model.forward(ld.LANL_Data.TRAIN)
            loss = model.loss_fn(zs, ld.LANL_Data.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, [loss])
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            print('[%d] Loss %0.4f  %0.2fs' % (e, loss.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(ld.LANL_Data.TRAIN, no_grad=True)
            v_loss = model.loss_fn(zs, ld.LANL_Data.VAL).item()

            print("\t Val loss: %0.4f" % v_loss)

            if v_loss > best[1]:
                best = (model.save_states(), v_loss)
            else:
                if e >= kwargs['min']:
                    no_progress += 1 

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break 

    model.load_states(best[0][0], best[0][1])
    zs, h0 = model(ld.LANL_Data.TEST, include_h=True)

    states = {'gcn': best[0][0], 'rnn': best[0][1]}
    f = open('model_save.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % (sum(times)/len(times)) )
    
    return model, zs[-1], h0


def test(model, zs, h0, rrefs):
    # Load train data into workers
    ld_args = get_work_units(len(rrefs), TE_START, TE_END, DELTA, True)
    
    print("Loading test data")
    futs = [
        _remote_method_async(
            GAE_DDP.load_new_data,
            rrefs[i], 
            ld.load_lanl_dist, 
            ld_args[i]
        ) for i in range(len(rrefs))
    ]

    # Wait until all workers have finished
    [f.wait() for f in futs]

    with torch.no_grad():
        model.eval()
        zs = model(ld.LANL_Data.TEST, h_0=h0, no_grad=True)

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    scores, labels, readable = model.score_all(zs, detailed=True)

    print("Sorting")
    sorted = scores.argsort()

    scores = scores[sorted]
    labels = labels[sorted]
    readable = [readable[i] for i in sorted]

    tot_anoms = labels.sum().item()

    # Classify using cutoff from earlier
    classified = torch.zeros(labels.size())
    classified[labels <= model.cutoff] = 1
    
    n_anoms = 0
    n_edges = scores.size(0)
    with open('out.txt', 'w+') as f:
        for i in (labels == 1).nonzero():    
            n_anoms += 1
            tpr = n_anoms / tot_anoms 
            fpr = (i-n_anoms) / n_edges 
            stats = 'TPR: %0.4f  FPR: %0.4f' % (tpr, fpr)

            f.write('[%d/%d] %0.4f %s  %s\n' % 
                (i, n_edges, scores[i], readable[i], stats)
            )

    tpr = classified[labels==1].mean() * 100
    fpr = classified[labels==0].mean() * 100
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))

    with open('out.txt', 'a') as f:
        f.write("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))


if __name__ == '__main__':
    max_workers = (TR_END-TR_START) // DELTA 
    workers = min(max_workers, WORKERS)

    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )