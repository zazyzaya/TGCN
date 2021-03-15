from copy import deepcopy

import torch 
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, \
    average_precision_score, roc_auc_score

import generators as g
import loaders.load_cic as lc 
from models.serial_model import SerialTGCN
from models.vgrnn_like import VGRNN
from utils import get_score

torch.set_num_threads(16)

LR = 0.001
PATIENCE = 25

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

'''
Given a list of z's and ei's return a stack of 
[src_embed + dst_embed]
For each time slice combined
'''
def cat_embeds(z, ei, decode='dot'):
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

def fmt_cm(cm):
    spacing = [
        len(str(cm[i][0]))
        for i in range(2)
    ]

    cell_size = max(spacing)
    pad1 = cell_size-spacing[0]
    pad2 = cell_size-spacing[1]
    
    print("   PP%s  PN" % (' '* (cell_size-1)))
    print("TP %d%s | %d" % (cm[0][0], ' '*pad1, cm[0][1]))
    print("TN %d%s | %d" % (cm[1][0], ' '*pad2, cm[1][1]))


def train(model, data, epochs=1500, nratio=1, dynamic=True, 
        min_epochs=100, lr_nratio=1, single_prior=False):
    TE_STARTS = data.te_starts

    opt = Adam(model.parameters(), lr=LR)
    best = (0, None)
    no_improvement = 0

    for e in range(epochs):
        model.train()
        opt.zero_grad()

        zs = model(data.x, data.eis[:TE_STARTS], data.tr)
        
        # TGCN uses the embeds of timestep t to predict t+1 if dynamic, using sparse
        # loss, requiring us to generate some neg samples, and timeshift them
        if model.__class__ == SerialTGCN:
            if dynamic:
                p,n,z = g.dynamic_link_prediction(
                    data, data.tr, zs, end=TE_STARTS, 
                    nratio=nratio, include_tr=False
                )
            else:
                p,n,z = g.link_prediction(
                    data, data.tr, zs, end=TE_STARTS, 
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
            p,n,z = g.link_prediction(
                data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
            )
        else:
            zs = zs[:-1]
            p,n,z = g.dynamic_link_prediction(
                data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
            )
    else:
        zs = zs[1:]
        p,n,z = g.link_prediction(
            data, data.all, zs_all, end=TE_STARTS, nratio=lr_nratio
        )

    # Use logistic regression to determine edge classification given likelihood
    # Train on known pos and neg edges from training set time slices
    lr = LogisticRegression(max_iter=1000)
    X_pos = cat_embeds(z, p)
    X_neg = cat_embeds(z, n)

    X_tr = torch.cat([X_pos, X_neg], dim=0)
    y_tr = torch.zeros(X_pos.size(0)+X_neg.size(0))
    y_tr[:X_pos.size(0)] = 1

    lr.fit(X_tr, y_tr)

    likelihood = [
        model.decode(data.eis[TE_STARTS+i][0], data.eis[TE_STARTS+i][1], zs[i])
        for i in range(zs.size(0))
    ]
    likelihood = torch.cat(likelihood, dim=0).unsqueeze(-1)
    y = torch.cat(data.y[TE_STARTS:], dim=0).squeeze(-1)

    likelihood = likelihood.squeeze(-1).numpy()
    ap = average_precision_score(y, likelihood)
    auc = roc_auc_score(y, likelihood)

    X_te = cat_embeds(zs, data.eis[TE_STARTS:])
    y_hat = lr.predict(X_te)

    idx = 0
    for i in range(TE_STARTS, data.T):
        ne = data.eis[i].size(1)
        yt = y[idx:idx+ne]
        y_hatt = y_hat[idx:idx+ne]

        print(data.times[i])
        fmt_cm(confusion_matrix(yt, y_hatt))
        print()
        idx += ne

    cm = confusion_matrix(y, y_hat)
    cr = classification_report(y, y_hat)

    print(cr)
    fmt_cm(cm)
    print('\n%s' % fmt_score([auc, ap]))


if __name__ == '__main__':
    pred=True
    data = lc.load_cic(delta=2)

    m = VGRNN(data.x.size(1), 32, 16, pred=pred, adj_loss=False)
    '''
    m = SerialTGCN(
        data.x.size(1), 128, 64, 
        variational=True, gru_hidden_units=2
    )
    '''

    train(
        m, data, dynamic=pred, nratio=1, 
        min_epochs=50, lr_nratio=5, single_prior=False
    )