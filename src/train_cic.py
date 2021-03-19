from copy import deepcopy

import torch 
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

import generators as g
import loaders.load_cic as lc 
from link_prediction import LP_Classifier
from models.serial_model import SerialTGCN, SerialTGCNGraphGRU
from models.vgrnn_like import VGRNN
from utils import get_score

torch.set_num_threads(16)

LR = 0.001
PATIENCE = 25
WINDOW = 4

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

'''
Given a list of z's and ei's return a stack of 
[src_embed + dst_embed]
For each time slice combined
'''
def cat_embeds(z, ei, decode='cat'):
    if decode=='dot':
        catted = [
            torch.sigmoid(
                (z[i][ei[i][0]] * z[i][ei[i][0]]).sum(dim=1)
            ).unsqueeze(-1)
            for i in range(len(ei))
        ]

    elif decode=='avg':
        catted = [    
            (z[i][ei[i][0]] + z[i][ei[i][0]]).true_divide(2.0)
            for i in range(len(ei))
        ]

    else:
        catted = [
            torch.cat([z[i][ei[i][0]], z[i][ei[i][0]]], dim=1)
            for i in range(len(ei))
        ]

    return torch.cat(catted, dim=0)


def train(model, data, epochs=1500, nratio=1, dynamic=True, 
        min_epochs=100, lr_nratio=1, single_prior=False, 
        lp_epochs=100, no_test=False):

    TE_STARTS = data.T if no_test else data.te_starts
    WINDOW = TE_STARTS

    opt = Adam(model.parameters(), lr=LR)
    best = (0, None)
    no_improvement = 0

    for e in range(epochs):
        model.train()
        opt.zero_grad()

        st = 0 #e % (max(1, TE_STARTS-WINDOW))
        zs = model(data.x, data.eis[st:st+WINDOW], data.tr, start_idx=st)
        
        # TGCN uses the embeds of timestep t to predict t+1 if dynamic, using sparse
        # loss, requiring us to generate some neg samples, and timeshift them
        if model.__class__ == SerialTGCN:
            if dynamic:
                p,n,z = g.dynamic_link_prediction(
                    data, data.tr, zs, end=st+WINDOW, start=st,
                    nratio=nratio, include_tr=False
                )
            else:
                p,n,z = g.link_prediction(
                    data, data.tr, zs, start=st, end=st+WINDOW, 
                    nratio=nratio, include_tr=False
                )
        
        # VGRNN uses dense loss, so no need to do neg sampling or timeshift
        else:
            if model.adj_loss:
                p = data.eis[:TE_STARTS]
                n = None
                z = zs 
            else:
                p,n,z = g.link_prediction(
                    data, data.tr, zs, end=TE_STARTS, 
                    nratio=nratio, include_tr=False
                )

        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis[:TE_STARTS], data.tr)
        
            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs, end=TE_STARTS)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[1]

            else:
                # VGRNN is providing priors, which are built from the previous timestep
                # already, thus there is no need to shift the selected ei's as the 
                # dynamic functions do 
                if model.__class__ == VGRNN:
                    zs = zs
                    dp,dn,dz = g.link_prediction(data, data.va, zs, end=TE_STARTS)
                else:
                    dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs, end=TE_STARTS)
                
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, data.all, zs, end=TE_STARTS)
                
                # Again, we don't need to shift the VGRNN embeds backward
                if model.__class__ == VGRNN:
                    dz = zs 
                
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = (
                    dscores[0] + dscores[1] +
                    dnscores[0] + dnscores[1]
                )
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > min_epochs:
                    no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break

    model = best[1]
    if no_test:
        return model
    
    zs = None
    with torch.no_grad():
        model.eval()
        if model.__class__ == SerialTGCN:
            zs_all, h0 = model(data.x, data.eis[:TE_STARTS], data.all, include_h=True)
        else:
            zs_all = model(data.x, data.eis[:TE_STARTS], data.all)

        # Generate all future embeds using prior from last normal state
        if single_prior:
            zs = torch.cat([
                model(data.x, [data.eis[i]], data.all, h_0=h0, start_idx=i)
                for i in range(TE_STARTS-1, data.T)
            ], dim=0)
        else:
            zs = model(data.x, data.eis, data.all)[TE_STARTS-1:]

    if dynamic: 
        if model.__class__ == VGRNN:
            zs = zs[1:]
        else:
            zs = zs[:-1]
    else:
        zs = zs[1:]
    
    # Train the link classifier
    lp = LP_Classifier()
    tr_mask = None
    X_pos_tr = None
    X_pos_va = None
    neg_val_size = None
    for _ in range(lp_epochs):
        if dynamic:
            if model.__class__ == VGRNN:
                p,n,z = g.link_prediction(
                    data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
                )
            else:
                p,n,z = g.dynamic_link_prediction(
                    data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
                )
        else:
            p,n,z = g.link_prediction(
                data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
            )

        X_neg = cat_embeds(z, n)

        # Only need to do this once. Don't know the size of the tr set
        # until now. But partitions don't change 
        if type(tr_mask) == type(None):
            X_pos = cat_embeds(z, p)

            tr_mask = torch.zeros(X_pos.size(0), dtype=torch.bool)
            prm = torch.randperm(X_pos.size(0))

            val_size = int(X_pos.size(0) * 0.05)
            tr_mask[prm[val_size:]] = True
            X_pos_tr, X_pos_va = X_pos[tr_mask], X_pos[~tr_mask]
            neg_val_size = val_size * lr_nratio

        # This is recalculated each time, because sending different randomly
        # generated negative samples each epoch
        X_neg_tr, X_neg_va = X_neg[neg_val_size:], X_neg[:neg_val_size]

        X_tr = torch.cat([X_pos_tr, X_neg_tr], dim=0)
        y_tr = torch.zeros(X_pos_tr.size(0)+X_neg_tr.size(0))
        y_tr[:X_neg_tr.size(0)] = 1

        X_va = torch.cat([X_pos_va, X_neg_va], dim=0)
        y_va = torch.zeros(val_size+neg_val_size)
        y_va[:neg_val_size] = 1

        lp.train_lp_step(X_tr, y_tr, X_va, y_va)

    likelihood = [
        model.decode(data.eis[TE_STARTS+i][0], data.eis[TE_STARTS+i][1], zs[i])
        for i in range(zs.size(0))
    ]
    likelihood = torch.cat(likelihood, dim=0)

    # Statistical measures on likelihood scores
    y = torch.cat(data.y[TE_STARTS:], dim=0).squeeze(-1)
    ap = average_precision_score(y, likelihood)
    auc = roc_auc_score(y, likelihood)

    X_te = cat_embeds(zs, data.eis[TE_STARTS:])
    lp.score(y, lp(X_te))
    lp.dumb_predict(likelihood, y)

    print('\n%s' % fmt_score([auc, ap]))


if __name__ == '__main__':
    pred=False
    #tr_data = lc.load_cic(delta=3, end='09:20', fname='M')
    #tr_data.x = torch.eye(3021)

    #m = VGRNN(data.x.size(1), 32, 16, pred=pred, adj_loss=False)
    data = lc.load_cic(fname='H0', end='10:00', delta=3)#, node_map=nm)
    m = SerialTGCN(
        data.x.size(1), 32, 16, use_graph_gru=False,
        variational=True, gru_hidden_units=1, use_predictor=False
    )

    '''
    # Pretrain on clean data
    m = train(
        m, tr_data, dynamic=pred, nratio=1, epochs=25,
        min_epochs=0, lr_nratio=1, single_prior=False,
        no_test=True
    )

    nm = tr_data.node_map
    print(len(nm))
    del tr_data 
    '''

    print(len(data.node_map))
    train(
        m, data, dynamic=pred, nratio=1,# epochs=5,
        min_epochs=10, lr_nratio=1, single_prior=False
    )