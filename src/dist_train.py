import os
import pickle 

import torch 
from torch import nn
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

import generators as g
import loaders.load_lanl_dist as ld 
from models.distributed.utils import _remote_method
from models.distributed.tgcn import TGCN, MyDDP, get_remote_gae
from utils import get_score, tpr_fpr

DEFAULTS = {
    'h_size': 32, 
    'z_size': 16,
    'lr': 0.01,
    'epochs': 1500,
    'min': 50,
    'patience': 50,
    'variational': True,
    'nratio': 10
}

DELTA=1000
START=0
END=ld.DATE_OF_EVIL_LANL-DELTA

def init_workers(num_workers, h_dim, start, end, delta, isTe):
    rrefs = []
    
    slices_needed = (end-start) // delta
    slices_needed += 1

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    # Put remaining tasks on last workers since it's likely the 
    # final timeslice is stopped halfway (ie it's less than a delta
    # so giving it extra timesteps is more likely okay)
    for i in range(num_workers, slices_needed%num_workers, -1):
        per_worker[i-1]+=1 

    prev = start
    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            ld_kwargs = {
                'start': prev, 
                'end': end_t,
                'delta': delta, 
                'is_test': isTe
            }
            prev = end_t

            rrefs.append(
                rpc.remote(
                    'worker'+str(i),
                    get_remote_gae,
                    args=(h_dim, ld.load_partial_lanl, ld_kwargs),
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
        # Master gets 16 threads and 4x4 threaded workers
        # In theory, only 16 threads should run at a time while
        # master sleeps, waiting on worker procs
        torch.set_num_threads(16)

        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        if 'variational' in tr_args:
            gae_kwds = {'variational': tr_args['variational']}
        else:
            gae_kwds = {}

        rrefs = init_workers(world_size-1, tr_args['h_size'], START, END, DELTA, False)
        gcn, rnn = train(rrefs, **tr_args)
        
        # TODO load workers with test data first
        #test(rrefs, gcn, rnn)

    # Slaves
    else:
        # If there are 4 workers, give them each 4 threads 
        # (Total 16 is equal to serial model)
        torch.set_num_threads(4)
        
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
    model = TGCN(rrefs, kwargs['h_size'], kwargs['z_size'])
    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    # Generate edge lists for loss calculation
    print("Getting training edges")
    tr_edges = [
        _remote_method(
            MyDDP.get_eis,
            rref,
            ld.LANL_Data.TRAIN
        ) for rref in rrefs
    ]
    tr_edges = sum(tr_edges, [])

    print("Getting validation edges")
    va_edges = [
        _remote_method(
            MyDDP.get_eis,
            rref,
            ld.LANL_Data.TRAIN
        ) for rref in rrefs
    ]
    va_edges = sum(va_edges, [])
    num_nodes = _remote_method(MyDDP.get_x_dim, rrefs[0])

    best = (None, 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        model.train()
        with dist_autograd.context() as context_id:
            zs = model.forward(ld.LANL_Data.TRAIN)
            n = g.lightweight_lp(tr_edges, num_nodes, nratio=kwargs['nratio'])
            loss = model.loss_fn(tr_edges, n, zs)

            print("backward")
            dist_autograd.backward(context_id, [loss])
            
            print("step")
            opt.step(context_id)

            print('[%d] Loss %0.4f' % (e, loss.item()))

        # TODO put with torch.no_grad() in the dist models 
        # as using it here only disables it for the master's model
        model.eval()
        with torch.no_grad():
            zs = model.forward(ld.LANL_Data.TRAIN)
            n = g.lightweight_lp(
                tr_edges, num_nodes, 
                num_pos=[
                    va_edges[i].size(1) for i in range(len(va_edges))
                ]
            )

            p,n = model.score_fn(va_edges,n,zs)
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

    return best[0]

def test(rrefs, data, gcn, rnn, **kwargs):
    model = TGCN(rrefs, data.x.size(1), kwargs['h_size'], kwargs['z_size'], variational=False)
    model.load_states(gcn, rnn)

    SKIP = data.te_starts 
    with torch.no_grad():
        model.eval()
        zs = model(data.x, data.eis, data.all, ew_fn=data.all_w)[SKIP:]

    # Scores all edges and matches them with name/timestamp
    edges = []
    data.node_map = pickle.load(open(ld.LANL_FOLDER+'nmap.pkl', 'rb'))
    
    for i in range(zs.size(0)):
        idx = i + data.te_starts

        ei = data.eis[idx]
        scores = model.decode(ei[0], ei[1], zs[i])
        names = data.format_edgelist(idx)

        for i in range(len(names)):
            edges.append(
                (scores[i].item(), names[i])
            )

    max_anom = (0, 0.0)
    edges.sort(key=lambda x : x[0])
    anoms = 0
    
    with open('out.txt', 'w+') as f:
        for i in range(len(edges)):
            e = edges[i]
            
            if 'ANOM' in e[1]:
                anoms += 1
                max_anom = (i, e[0])
                stats = tpr_fpr(i, anoms, len(edges), data.tot_anoms)
                f.write('[%d/%d] %0.4f %s  %s\n' % (i, len(edges), e[0], e[1], stats))

    print(
        'Maximum anomaly scored %d out of %d edges'
        % (max_anom[0], len(edges))
    )


if __name__ == '__main__':
    world_size = 5

    mp.spawn(
        init_procs,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )