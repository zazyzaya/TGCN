from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace as SN

import pandas as pd
import torch 
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.loop import add_remaining_self_loops

import generators as g
import loaders.load_vgrnn as vd
from models.tgcn_prob_loss import ProbTGCN, HybridProbTGCN
from utils import get_score, tf_auprc

torch.set_num_threads(8)

NUM_TESTS = 5

PATIENCE = 100
MAX_DECREASE = 2

TEST_TS = 3

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(model, data, epochs=1500, dynamic=False, nratio=10, lr=0.01):
    print(lr)
    end_tr = data.T-TEST_TS

    opt = Adam(model.parameters(), lr=lr)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        zs = None

        # Get embedding        
        zs, _ = model(data.x, data.eis[:end_tr], data.tr)

        if not dynamic:
            p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, nratio=nratio)
            
        else:
            p,n,z = g.dynamic_link_prediction(data, data.tr, zs, include_tr=False, nratio=nratio)      
        
        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        trloss = loss.item() 
        with torch.no_grad():
            model.eval()
            zs, hs = model(data.x, data.eis[:end_tr], data.tr)

            if not dynamic:
                p,n,h = g.link_prediction(data, data.va, hs)
                st, sf = model.score_fn(p,n,h)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[0] + sscores[1]

            elif type(model) == HybridProbTGCN:
                p,n,_ = g.dynamic_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,hs)
                dscores = get_score(dt, df)

                dp,dn,_ = g.dynamic_new_link_prediction(data, data.va, zs)
                dt, df = model.score_fn(dp,dn,hs)
                dnscores = get_score(dt, df)

            else:
                dp,dn,dh = g.dynamic_link_prediction(data, data.va, hs)
                dt, df = model.score_fn(dp,dn,dh)
                dscores = get_score(dt, df)

                dp,dn,dh = g.dynamic_new_link_prediction(data, data.va, hs)
                dt, df = model.score_fn(dp,dn,dh)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = (
                    dscores[0] + dscores[1] 
                )

            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                no_improvement += 1
                if no_improvement == PATIENCE:
                    print("Early stopping...\n")
                    break

    # Test the data that hasn't been touched
    model = best[1]
    with torch.no_grad():
        model.eval()
        _,hs = model(data.x, data.eis, data.tr)
        hs = hs[end_tr:] if not dynamic else hs[end_tr-1:]

        if not dynamic and type(model) != HybridProbTGCN:
            p,n,h = g.link_prediction(data, data.te, hs, start=end_tr)
            t, f = model.score_fn(p,n,h)
            sscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Static LP:  %s
                '''
            % fmt_score(sscores))

            return {'auc': sscores[0], 'ap': sscores[1]}

        else:
            p,n,h = g.dynamic_link_prediction(data, data.te, hs, start=end_tr-1)
            print(len(p))
            print(h.size(0))
            t, f = model.score_fn(p,n,h)
            dscores = get_score(t, f)

            p,n,h = g.dynamic_new_link_prediction(data, data.te, hs, start=end_tr-1)
            t, f = model.score_fn(p,n,h)
            nscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Dynamic LP:     %s 
                    Dynamic New LP: %s 
                ''' %
                (fmt_score(dscores),
                 fmt_score(nscores))
            )

            return {
                'pred-auc': dscores[0],
                'pred-ap': dscores[1],
                'new-auc': nscores[0], 
                'new-ap': nscores[1],
            }


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument(
        '--lr',
        type=float,
        default=0.01
    )
    ap.add_argument(
        '-s', '--static',
        action='store_false'
    )
    args = ap.parse_args()

    outf = 'tedge_gru.txt' 
    for d in ['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d)
        model = ProbTGCN(data.x_dim, 32, 16, lstm=False)
        stats = [train(deepcopy(model), data, dynamic=args.static, lr=args.lr) for _ in range(NUM_TESTS)]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('===== LR %0.4f =====\n' % (args.lr))
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()