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
from distmodel.utils import _remote_method_async
from distmodel.tgcn import TGCN, GAE_DDP, get_remote_gae
from utils import get_score, tpr_fpr, get_optimal_cutoff

DEFAULTS = {
    'h_size': 64, 
    'z_size': 32,
    'lr': 0.001,
    'epochs': 1500,
    'min': 25,
    'patience': 5,
    'n_gru': 2,
    'variational': False,
    'nratio': 10,
    'pred': True
}

WORKERS=8
W_THREADS=1
M_THREADS=2

DELTA=(60**2) * 2
START=0
END=ld.DATE_OF_EVIL_LANL-1

TE_START=ld.DATE_OF_EVIL_LANL
TE_END=5011200 # Full  #1089597 # 500 anomalies
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

        rrefs = init_workers(world_size-1, tr_args['h_size'], START, END, DELTA, False)
        model, zs, h0 = train(rrefs, **tr_args)
        #test(model, zs, h0, rrefs, **tr_args)

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


def train(rrefs, **kwargs):
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
        model.train()
        with dist_autograd.context() as context_id:
            st = time.time()
            zs = model.forward(ld.LANL_Data.TRAIN)
            loss = model.loss_fn(zs, nratio=kwargs['nratio'], pred=kwargs['pred'])

            print("backward")
            dist_autograd.backward(context_id, [loss])
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            print('[%d] Loss %0.4f  %0.2fs' % (e, loss.item(), elapsed))

        # TODO put with torch.no_grad() in the dist models 
        # as using it here only disables it for the master's model
        model.eval()
        with torch.no_grad():
            zs = model.forward(ld.LANL_Data.TRAIN, no_grad=True)
            p,n = model.score_fn(zs, pred=kwargs['pred'])
            auc, ap = get_score(p,n)

            print("\tAUC: %0.4f  AP: %0.4f" % (auc, ap))

            val_perf = auc+ap 
            if val_perf > best[1]:
                best = (model.save_states(), val_perf)
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
    
    get_cutoff(model, **kwargs)
    return model, zs[-1], h0

def get_cutoff(model, **kwargs):
    with torch.no_grad():
        model.eval()
        zs = model.forward(ld.LANL_Data.ALL, no_grad=True)
        p,n = model.score_fn(zs, pred=kwargs['pred'])

    model.cutoff = get_optimal_cutoff(p,n)
    return model.cutoff

def test(model, zs, h0, rrefs, **kwargs):
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

    '''
    # Combines all subgraphs into one graph on rrefs[0]
    _remote_method(
        GAE_DDP.student_becomes_master,
        rrefs[0],
        rrefs[1:],
        ld.reducer
    )

    # Only need one GCN now
    model.gcns = [rrefs[0]]
    model.num_workers = 1

    # Hopefully GC will get the rest..
    rrefs = rrefs[0]
    '''

    with torch.no_grad():
        model.eval()
        zs = model(ld.LANL_Data.TEST, h_0=h0, no_grad=True)

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    edges, tot_anoms = model.score_edges(zs, detailed=True)
    max_anom = (0, 0.0)

    print("Sorting")
    edges.sort(key=lambda x : x[0])
    anoms = 0
    
    y, y_hat = [],[]
    with open('out.txt', 'w+') as f:
        for i in range(len(edges)):
            e = edges[i]
            
            if e[1] == 1 or (type(e[1]) == str and 'ANOM' in e[1]):
                anoms += 1
                max_anom = (i, e[0])
                stats = tpr_fpr(i, anoms, len(edges), tot_anoms)
                f.write('[%d/%d] %0.4f %s  %s\n' % (i, len(edges), e[0], e[1], stats))

                y.append(1)
            else:
                y.append(0)

            y_hat.append(e[0])

    # This is such a stupid way to do this. Maybe find a better one for the future
    y = np.array(y)
    y_hat = np.array(y_hat)
    y_hat[y_hat > model.cutoff] = 0
    y_hat[y_hat <= model.cutoff] = 1

    tpr = y_hat[y==1].mean() * 100
    fpr = y_hat[y==0].mean() * 100
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))

    with open('out.txt', 'a') as f:
        f.write("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))

    print(
        'Maximum anomaly scored %d out of %d edges'
        % (max_anom[0], len(edges))
    )

if __name__ == '__main__':
    max_workers = (END-START) // DELTA 
    workers = min(max_workers, WORKERS)

    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )