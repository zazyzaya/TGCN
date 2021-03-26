from copy import deepcopy
import pickle 

import torch
from torch.optim import Adam

import generators as g
import loaders.load_cyber as lc 
import loaders.load_lanl_dist as ld
from models.serial_model import SerialTGCN
from utils import get_score

torch.set_num_threads(16)
fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

LR=0.001
PATIENCE=50

def tpr_fpr(rank, n, total, tot_anom):
    # TPR is easy
    tpr = n/rank

    # FPR is trickier 
    fp = rank-n
    tn = total-rank-tot_anom
    fpr = fp / (fp+tn)

    return "TPR: %0.4f, FPR: %0.4f" % (tpr*100, fpr*100)

def train(data, model, dynamic, epochs=1500):
    # Leave all params as default for now
    SKIP = data.te_starts 
    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embeddings
        zs = model(data.x, data.eis[:SKIP], data.tr, ew_fn=data.tr_w)
        
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
            zs = model(data.x, data.eis[:SKIP], data.tr, ew_fn=data.tr_w)
        
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
    return model 

def test(data, model, pred, single_prior=False):
    SKIP = data.te_starts 
    with torch.no_grad():
        model.eval()
        zs, h = model(data.x, data.eis, data.all, ew_fn=data.all_w, include_h=True)[:data.te_starts]

        # Generate all future embeds using prior from last normal state
        if single_prior:
            zs = torch.cat([
                model(data.x, [data.eis[i]], data.all, ew_fn=data.all_w, h_0=h, start_idx=i)
                for i in range(SKIP-1, data.T)
            ], dim=0)
        else:
            zs = model(data.x, data.eis, data.all, ew_fn=data.all_w)[SKIP-1:]

    zs = zs[:-1] if pred else zs[1:] 

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
    HOME = '/mnt/raid0_24TB/isaiah/code/TGCN/src/'
    pred = False
    te_start = ld.DATE_OF_EVIL_LANL-1000

    # Load training data and train
    data = ld.load_partial_lanl(start=0, is_test=False)
    model = SerialTGCN(
        data.x.size(1), 32, 16,
        variational=True, gru_hidden_units=2
    )
    model = train(data, model, pred)

    # Load testing data and test
    model = torch.load(HOME + '../pretrained/pretrained_LANL_2GRU.model')
    data = ld.load_partial_lanl(start=0, is_test=True)
    test(
        data, model, pred, single_prior=False
    )

    torch.save(model, HOME + 'pretrained_LANL.model')