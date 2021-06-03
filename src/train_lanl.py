import argparse
from copy import deepcopy
import json 
import time 
import pickle

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

import generators as g
import loaders.load_lanl_dist as ld
from models.serial_model import SerialTGCN
from models.node2vec import embed
from utils import get_score, get_optimal_cutoff

torch.set_num_threads(16)
fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

HOME = '/mnt/raid0_24TB/isaiah/code/TGCN/src/'
MODELS = '/mnt/raid0_24TB/isaiah/code/TGCN/pretrained/'

LR=0.001
PATIENCE=5
EPOCHS=1500

TR_START=0
TR_END=ld.DATE_OF_EVIL_LANL-1

TE_START=TR_END
#TE_END = 228642 # First 20 anoms
#TE_END = 740104 # First 100 anoms
#TE_END = 1089597 # First 500 anoms
TE_END = 5011199 # Full

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-d', '--delta',
        type=float, default=4
    )
    p.add_argument(
        '-t', '--tdelta',
        type=float, default=-1
    )
    p.add_argument(
        '-l', '--load',
        type=str, default=''
    )

    p.add_argument(
        '-s', '--single',
        action='store_true'
    )

    p.add_argument(
        '-v', '--save',
        action='store_true'
    )

    p.add_argument(
        '-g', '--grus',
        type=int, default=1
    )

    p.add_argument(
        '--hidden',
        type=int, default=32
    )

    p.add_argument(
        '-e', '--embed',
        type=int, default=16
    )

    p.add_argument(
        '-p', '--predictive',
        action='store_true'
    )

    p.add_argument(
        '-n', '--node2vec',
        action='store_true'
    )

    cleaned = {}
    args = p.parse_args()
    #args = p.parse_args('-l 2hr -d 2'.split(' '))

    cleaned['single'] = args.single
    cleaned['save'] = args.save
    cleaned['grus'] = args.grus
    cleaned['hidden'] = args.hidden
    cleaned['embed'] = args.embed
    cleaned['predictive'] = args.predictive
    cleaned['node2vec'] = args.node2vec
    
    # Get delta and test delta
    cleaned['delta'] = args.delta 
    if args.tdelta <= 0:
        cleaned['tdelta'] = args.delta 
    else:
        cleaned['tdelta'] = args.tdelta

    # Manage file names
    if args.load:
        cleaned['train'] = False 
        cleaned['load'] = args.load 
    else:
        cleaned['train'] = True 
        cleaned['load'] = str(cleaned['delta']) + 'hr'

    # Convert to seconds 
    cleaned['delta'] *= 60**2
    cleaned['tdelta'] *= 60**2 

    cleaned['delta'] = int(cleaned['delta'])
    cleaned['tdelta'] = int(cleaned['tdelta'])

    return cleaned


def train(data, model, dynamic, epochs=1500, nratio=10):
    # Leave all params as default for now
    opt = Adam(model.parameters(), lr=LR)

    times = []

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        start = time.time()
        
        # Generate positive and negative samples from this and the next time step
        if dynamic:
            zs = model(data.x, data.eis, data.all, ew_fn=data.all_w)
            p,n,z = g.dynamic_link_prediction(
                data, data.tr, zs, 
                include_tr=False,
                nratio=nratio
            )
        else:
            zs = model(data.x, data.eis, data.tr, ew_fn=data.tr_w)
            p,n,z = g.link_prediction(
                data, data.tr, zs, 
                include_tr=False,
                nratio=nratio
            )
        
        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        elapsed = time.time() - start
        times.append(elapsed)

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            if not dynamic:
                zs = model(data.x, data.eis, data.tr, ew_fn=data.tr_w)
                p,n,z = g.link_prediction(data, data.va, zs)
                sp, sf = model.score_fn(p,n,z)
                sscores = get_score(sp,sf)
                #vloss = model.loss_fn(p,n,z).item()

                print(
                    '[%d] Loss: %0.4f \t%0.4fs  \n\tDet %s\n' %
                    (e, trloss, elapsed, fmt_score(sscores))
                )

                avg = sum(sscores)

            else:
                zs = model(data.x, data.eis, data.all, ew_fn=data.all_w)
                dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                #dp,dn,dz = g.dynamic_new_link_prediction(data, data.va, zs)
                #dt, df = model.score_fn(dp,dn,dz)
                #dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f \t%0.4fs  \n\tPred %s ' % #\n\tNew %s\n' %
                    (e, trloss, elapsed, fmt_score(dscores))#, fmt_score(dnscores) )
                )

                avg = sum(dscores)
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:    
                no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break

    print("Avg. TPE: %0.4fs" % (sum(times)/len(times)) )
    model = best[1]
    _, h0 = model(data.x, data.eis, data.all, ew_fn=data.all_w, include_h=True)

    return model, h0

'''
After model is trained, run on train data to find best cutoff point
to mark as anomalous
'''
def get_cutoff(data, model, pred, h0=None, nratio=10):
    with torch.no_grad():
        zs, h0 = model(data.x, data.eis, data.all, ew_fn=data.all_w, h_0=h0, include_h=True)

        # Get optimal cutoff point for LR
        if pred:
            p,n,z = g.dynamic_link_prediction(
                data, data.all, zs, 
                include_tr=False, nratio=nratio
            )
        else:
            p,n,z = g.link_prediction(
                data, data.all, zs,
                include_tr=False, nratio=nratio
            )

        dt, df = model.score_fn(p,n,z)
        
    model.cutoff = get_optimal_cutoff(dt, df, fw=0.6)
    return model.cutoff, h0


def test(data, model, h0, pred, single_prior=False, fname='out'):
    with torch.no_grad():
        model.eval()

        # Generate all future embeds using prior from last normal state
        if single_prior:
            zs = torch.cat([
                model(data.x, [data.eis[i]], data.all, ew_fn=data.all_w, h_0=h0, start_idx=i)
                for i in range(data.T)
            ], dim=0)

        else:
            zs = model(data.x, data.eis, data.all, ew_fn=data.all_w, h_0=h0)
        

    zs = zs[:-1] if pred else zs
    future = 1 if pred else 0

    # Scores all edges and matches them with name/timestamp
    y = []
    y_hat = []

    edges = []
    for i in range(zs.size(0)):
        ei = data.eis[i+future]
        scores = model.decode(ei[0], ei[1], zs[i])
        #names = data.format_edgelist(i+future)

        y += data.ys[i+future]
        y_hat.append(scores.squeeze(-1))

        '''
        for i in range(len(names)):
            edges.append(
                (scores[i].item(), names[i])
            )
        '''

    max_anom = (0, 0.0)
    edges.sort(key=lambda x : x[0])
    anoms = 0
    
    # Compute TPR/FPR according to precalculated cutoff
    y_hat = torch.cat(y_hat, dim=0).float().numpy()
    y = np.array(y)
    guesses = np.zeros(y_hat.shape)
    default = np.zeros(y_hat.shape)

    guesses[y_hat <= model.cutoff] = 1
    default[y_hat <= 0.51] = 1
    y_hat = guesses
    
    tpr = y_hat[y==1].mean() * 100
    fpr = y_hat[y==0].mean() * 100
    
    tp = y_hat[y==1].sum()
    fp = y_hat[y==0].sum()

    print("Opt. TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))

    '''
    tot_anoms = y.sum()
    with open(fname+'.txt', 'w+') as f:
        f.write("Using default cutoff 0.51:\n")
        f.write("TPR: %0.2f, FPR: %0.2f\n\n" % (d_tpr, d_fpr))

        f.write("Using learned cutoff %0.4f:\n" % cutoff)
        f.write("TPR: %0.2f, FPR: %0.2f\n\n" % (tpr, fpr))
        for i in range(len(edges)):
            e = edges[i]
            
            if 'ANOM' in e[1]:
                anoms += 1
                max_anom = (i, e[0])
                stats = tpr_fpr(i, anoms, len(edges), tot_anoms)
                f.write('[%d/%d] %0.4f %s  %s\n' % (i, len(edges), e[0], e[1], stats))

    print(
        'Maximum anomaly scored %d out of %d edges'
        % (max_anom[0], len(edges))
    )
    '''

    return (tpr, fpr, tp, fp)

'''
Spins up a model and runs it once given user's input args
'''
def run_once(args):
    val = (TR_END - TR_START) // 20
    VAL_START = TR_END-val
    VAL_END = TR_END

    dim_str = '_%dh_%de' % (args['hidden'], args['embed']) \
            if args['hidden'] != 32 or args['embed'] != 16 \
            else ''

    p_str = '_static' if not args['predictive'] else ''
    fname = '%sLANL_%s_%dgru%s%s.pkl' % (MODELS, args['load'], args['grus'], dim_str, p_str)

    pred = args['predictive']

    # Load training data and train
    if args['train']:
        data = ld.load_lanl_dist(
            min(8, (VAL_START-TR_START)//args['delta']), 
            start=TR_START, end=VAL_START,
            delta=args['delta']
        )

        if args['node2vec']:
            x = embed(200, data, 64, 10, 5)

        else:
            x = torch.eye(data.x.size(0))
    
        x = torch.cat(
                [data.x, x], dim=1
            )
        data.x = x 
        model = SerialTGCN(
            data.x.size(1), 32, 16,
            gru_hidden_units=args['grus'], use_w=True, 
            use_predictor=False, #neg_weight=0.75
        )
        model, h0 = train(data, model, pred, epochs=EPOCHS)

        # Get a bit more data for validation of optimal cutoff
        data = ld.load_lanl_dist(
            2, start=VAL_START, end=VAL_END,
            delta=args['delta']
        )
        data.x = x
        cutoff, h0 = get_cutoff(data, model, pred, h0)
        
        if args['save']:
            pickle.dump(
                {'model': model, 'h0': h0, 'x': data.x},
                open(fname, 'wb+')
            )

    # Load testing data and test
    else:
        sv = pickle.load(open(fname, 'rb'))
        model = sv['model']
        h0 = sv['h0']    
        x = sv['x']
        
        # Get a bit more data for validation of optimal cutoff
        data = ld.load_lanl_dist(
            min(8, (VAL_END-VAL_START)//args['delta']),
            start=VAL_START, end=VAL_END,
            delta=args['delta']
        )
        data.x = x
        cutoff, h0 = get_cutoff(data, model, pred, h0)
    
    data = ld.load_lanl_dist(
        min(16, (TE_END-TE_START)//args['tdelta']), 
        start=TE_START, end=TE_END, 
        is_test=True, delta=args['tdelta']
    )
    data.x = x 

    return test(
        data, model, h0, pred, fname=args['load'], single_prior=args['single']
    )

    
if __name__ == '__main__':
    args = get_args()
    print(json.dumps(args, indent=2))

    if args['train']:
        stats = [run_once(args) for _ in range(5)]
        stats = pd.DataFrame(stats, columns=['TPR', 'FPR', 'TP', 'FP'])
        stats = pd.DataFrame([stats.mean(), stats.sem()], index=['mean', 'std'])

        with open('lanl_stats.txt', 'a') as f:
            f.write(json.dumps(args, indent=2))
            f.write('\n'+str(stats)+'\n\n')

    else:
        run_once(args)
