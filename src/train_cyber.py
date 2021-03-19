from copy import deepcopy

import torch
from torch.optim import Adam

import generators as g
import loaders.load_cyber as lc 
from models.serial_model import SerialTGCN
from utils import get_score

torch.set_num_threads(16)
fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

LR=0.001
PATIENCE=50

def train_cyber(data, model, dynamic, single_prior=False, 
                epochs=1500):
    # Leave all params as default for now
    SKIP = data.te_starts 
    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embeddings
        zs = model(data.x, data.eis[:SKIP], data.tr)
        
        # Generate positive and negative samples from this and the next time step
        if dynamic:
            p,n,z = g.dynamic_link_prediction(
                data, data.tr, zs, 
                include_tr=False, end=SKIP
            )
        else:
            p,n,z = g.link_prediction(
                data, data.tr, zs, 
                include_tr=False, end=SKIP
            )
        
        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis[:SKIP], data.tr)
        
            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs, end=SKIP)
                sp, sf = model.score_fn(p,n,z)
                sscores = get_score(sp,sf)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s' %
                    (e, trloss, fmt_score(sscores))
                )

                avg = sum(sscores)

            else:
                dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs, end=SKIP)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, data.va, zs, end=SKIP)
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tPred %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = sum(dscores) + sum(dnscores)            
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:    
                no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break

    model = best[1]
    zs = None
    with torch.no_grad():
        model.eval()
        zs, h = model(data.x, data.eis, data.all, include_h=True)[:data.te_starts]

        # Generate all future embeds using prior from last normal state
        if single_prior:
            zs = torch.cat([
                model(data.x, [data.eis[i]], data.all, h_0=h, start_idx=i)
                for i in range(SKIP-1, data.T)
            ], dim=0)
        else:
            zs = model(data.x, data.eis, data.all)[SKIP-1:]

    zs = zs[:-1] if pred else zs[1:] 

    # Scores all edges and matches them with name/timestamp
    edges = []
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
    with open('out.txt', 'w+') as f:
        for i in range(len(edges)):
            e = edges[i]
            f.write('%0.4f %s\n' % e) 
            
            if 'ANOM' in e[1]:
                max_anom = (i, e[0])
                print('[%d/%d] %0.4f %s' % (i, len(edges), e[0], e[1]))

    print(
        'Maximum anomaly scored %d out of %d edges'
        % (max_anom[0], len(edges))
    )

    
if __name__ == '__main__':
    pred = False
    
    data = lc.load_lanl()
    model = SerialTGCN(
        data.x.size(1), 32, 16,
        variational=True, use_predictor=False,
        gru_hidden_units=2
    )

    train_cyber(data, model, pred, single_prior=False)