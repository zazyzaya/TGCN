from copy import deepcopy

import torch
from torch.optim import Adam

import generators as g
import loaders.load_cyber as lc 
from models.serial_model import SerialTGCN
from utils import get_score

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

LR=0.01
PATIENCE=50

def train_cyber(data, epochs=1500, te_history=0):
    # Leave all params as default for now
    SKIP = data.te_starts 
    model = SerialTGCN(
        data.x.size(1), 32, 16 
    )
    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embeddings
        zs = model(data.x, data.eis[:SKIP], data.tr)
        
        # Generate positive and negative samples from this and the next time step
        p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)
        dp,dn,dz = g.dynamic_link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)
        
        loss = model.loss_fn(p+dp, n+dn, torch.cat([z, dz], dim=0))
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis[:SKIP], data.tr)
        
            p,n,z = g.link_prediction(data, None, zs, end=SKIP)
            sp, sf = model.score_fn(dp,dn,dz)
            sscores = get_score(sp,sf)

            dp,dn,dz = g.dynamic_link_prediction(data, None, zs, end=SKIP)
            dt, df = model.score_fn(dp,dn,dz)
            dscores = get_score(dt, df)

            dp,dn,dz = g.dynamic_new_link_prediction(data, None, zs, end=SKIP)
            dt, df = model.score_fn(dp,dn,dz)
            dnscores = get_score(dt, df)

            print(
                '[%d] Loss: %0.4f  \n\tDet %s\n\tPred %s  \n\tNew %s' %
                (e, trloss, fmt_score(sscores), fmt_score(dscores), fmt_score(dnscores) )
            )

            avg = (
                sscores[0] + sscores[1]
            )
            
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
        zs = model(data.x, data.eis, data.all)[data.te_starts:]

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

    edges.sort(key=lambda x : x[0], reverse=True)
    for e in edges:
        print('%0.4f %s' % e) 
    
if __name__ == '__main__':
    data = lc.load_pico(delta=6)
    train_cyber(data)