import argparse
from copy import deepcopy
import json 
import time 
import pickle

import numpy as np
import torch
from torch.optim import Adam

import generators as g
import loaders.load_lanl_dist as ld
from models.serial_model import SerialTGCN
from utils import get_score, tpr_fpr, get_optimal_cutoff

torch.set_num_threads(16)
fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

LR=0.001
PATIENCE=5
EPOCHS=1500

T_END = 1089597 # First 500 anoms

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '-d', '--delta',
        type=int, default=4
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

    cleaned = {}
    args = p.parse_args()

    cleaned['single'] = args.single
    cleaned['save'] = args.save
    
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
        cleaned['load'] = str(int(cleaned['delta'])) + 'hr'

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

        # Get embeddings
        zs = model(data.x, data.eis, data.tr, ew_fn=data.tr_w)
        
        # Generate positive and negative samples from this and the next time step
        if dynamic:
            p,n,z = g.dynamic_link_prediction(
                data, data.tr, zs, 
                include_tr=False,
                nratio=nratio
            )
        else:
            p,n,z = g.link_prediction(
                data, data.all, zs, 
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
            zs = model(data.x, data.eis, data.tr, ew_fn=data.tr_w)
        
            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs)
                sp, sf = model.score_fn(p,n,z)
                sscores = get_score(sp,sf)

                print(
                    '[%d] Loss: %0.4f\t%0.4fs  \n\tDet %s\n' %
                    (e, trloss, elapsed, fmt_score(sscores))
                )

                avg = sum(sscores)

            else:
                dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                #dp,dn,dz = g.dynamic_new_link_prediction(data, data.va, zs)
                #dt, df = model.score_fn(dp,dn,dz)
                #dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f\t%0.4fs  \n\tPred %s ' % #\n\tNew %s\n' %
                    (e, trloss, elapsed, fmt_score(dscores))#, fmt_score(dnscores) )
                )

                avg = sum(dscores) #+ sum(dnscores)            
            
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
def get_cutoff(data, model, pred, nratio=10):
    with torch.no_grad():
        zs = model(data.x, data.eis, data.all, ew_fn=data.all_w)

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
        
    model.cutoff = get_optimal_cutoff(dt, df)
    return model.cutoff


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
        names = data.format_edgelist(i+future)

        y += data.ys[i+future]
        y_hat.append(scores.squeeze(-1))

        for i in range(len(names)):
            edges.append(
                (scores[i].item(), names[i])
            )

    max_anom = (0, 0.0)
    edges.sort(key=lambda x : x[0])
    anoms = 0
    
    # Compute TPR/FPR according to precalculated cutoff
    y_hat = torch.cat(y_hat, dim=0).numpy()
    y = np.array(y)
    y_hat[y_hat > model.cutoff] = 0
    y_hat[y_hat <= model.cutoff] = 1
    
    tpr = y_hat[y==1].mean() * 100
    fpr = y_hat[y==0].mean() * 100
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))

    tot_anoms = sum([sum(data.ys[i]) for i in range(data.T)])

    with open(fname+'.txt', 'w+') as f:
        f.write("Using learned cutoff %0.4f:" % cutoff)
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

    
if __name__ == '__main__':
    HOME = '/mnt/raid0_24TB/isaiah/code/TGCN/src/'
    MODELS = '/mnt/raid0_24TB/isaiah/code/TGCN/pretrained/'
    
    pred = True

    args = get_args()
    te_start = ld.DATE_OF_EVIL_LANL-1

    print(json.dumps(args, indent=2))

    # Load training data and train
    if args['train']:
        data = ld.load_lanl_dist(
            min(8, (te_start)//args['delta']), 
            start=0, end=te_start, 
            delta=args['delta']
        )
        model = SerialTGCN(
            data.x.size(1), 32, 16,
            variational=False, gru_hidden_units=1, use_w=True
        )
        model, h0 = train(data, model, pred, epochs=EPOCHS)
        cutoff = get_cutoff(data, model, pred)
        
        if args['save']:
            mdata = {
                'model': model,
                'h0': h0
            }

            pickle.dump(
                mdata, 
                open(MODELS + 'LANL_' + args['load'] + '.pkl', 'wb+'), 
                protocol=pickle.HIGHEST_PROTOCOL
            )


    # Load testing data and test
    else:
        mdata = pickle.load(open(MODELS + 'LANL_' + args['load'] + '.pkl', 'wb+'))
        model = mdata['model']
        h0 = mdata['h0']
        cutoff = model.cutoff
    
    data = ld.load_lanl_dist(
        min(8, (T_END-te_start)//args['tdelta']), 
        start=te_start, end=T_END, 
        is_test=True, delta=args['tdelta']
    )
    test(
        data, model, h0, pred, fname=args['load'], single_prior=args['single']
    )