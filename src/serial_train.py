from copy import deepcopy
import argparse
from numpy.lib.index_tricks import _fill_diagonal_dispatcher

import pandas as pd
import torch 
from torch.optim import Adam

import generators as g
import loaders.load_vgrnn as vd
from models.serial_model import SerialTGCN, SerialTGCNGraphGRU
from models.vgrnn_like import GAE_RNN, VGRNN
from models.tgcn_with_prior import PriorSerialTGCN
from utils import get_score

torch.set_num_threads(16)

uses_priors = [VGRNN, PriorSerialTGCN]

NUM_TESTS = 5

LR = 0.01
PATIENCE = 100
MAX_DECREASE = 2

KL_ANNEALING = 100
KL_WEIGHT = 0.0001

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(model, data, epochs=1500, dynamic=False, nratio=10):
    lr = LR

    # Test/Val on last 3 time steps
    SKIP = data.T-3
    decreases = 0
    print(LR)

    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        zs = None

        # Get embedding
        if dynamic:
            zs = model(data.x, data.eis[:SKIP], data.all)
            p,n,z = g.link_prediction(data, data.all, zs, include_tr=False, end=SKIP, nratio=nratio)
        else:
            zs = model(data.x, data.eis[:SKIP], data.tr)
            p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, end=SKIP, nratio=nratio)        

        if not dynamic:
            loss = model.loss_fn(p,n,z)
        
        elif dynamic and model.__class__ not in uses_priors:
            dp,dn,dz = g.dynamic_link_prediction(data, data.all, zs, include_tr=False, end=SKIP, nratio=nratio)      
            loss = model.loss_fn(dp, dn, dz)

        else:
            loss = model.loss_fn(p,n,z)
        

        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 
        with torch.no_grad():
            model.eval()
            if not dynamic:
                zs = model(data.x, data.eis, data.tr)[SKIP:]

                p,n,z = g.link_prediction(data, data.va, zs, start=SKIP)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[0] + sscores[1]

            else:
                zs = model(data.x, data.eis, data.all)[SKIP-1:]
                # VGRNN is providing priors, which are built from the previous timestep
                # already, thus there is no need to shift the selected ei's as the 
                # dynamic functions do 
                if model.__class__ in uses_priors:
                    zs = zs[1:]
                    dp,dn,dz = g.link_prediction(data, None, zs, start=SKIP)
                else:
                    dp,dn,dz = g.dynamic_link_prediction(data, None, zs, start=SKIP-1)
                
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, None, zs, start=SKIP-1)
                if model.__class__ in uses_priors:
                    dz = zs # Again, we don't need to shift the VGRNN embeds backward
                
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = (
                    dscores[0] + dscores[1] 
                    #+
                    #dnscores[0] + dnscores[1]
                )
            
            if e == KL_ANNEALING:
                model.kld_weight = KL_WEIGHT

            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > 100:
                    no_improvement += 1
                if no_improvement == PATIENCE:
                    # This doesn't improve anything, and I don't want to stray
                    # too far from the OG workflow the paper uses
                    if decreases < MAX_DECREASE and False:
                        decreases += 1 
                        model = best[1]
                        
                        lr /= 2
                        opt = Adam(model.parameters(), lr=lr)

                        no_improvement = 0
                        print("Lowering LR")

                    else:
                        print("Early stopping...\n")
                        break


    model = best[1]
    with torch.no_grad():
        model.eval()
        if not dynamic:
            _, h0 = model(data.x, data.eis[:SKIP], data.all, include_h=True)
            zs = model(data.x, data.eis[SKIP:], data.tr, h_0=h0, start_idx=SKIP)

            p,n,z = g.link_prediction(data, data.te, zs, start=SKIP)
            t, f = model.score_fn(p,n,z)
            sscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Static LP:  %s
                '''
            % fmt_score(sscores))

            return {'auc': sscores[0], 'ap': sscores[1]}

        else:
            zs = model(data.x, data.eis, data.all)[SKIP-1:]

            if model.__class__ in uses_priors:
                zs = zs[1:]
                p,n,z = g.link_prediction(data, None, zs, start=SKIP)
            else:                
                p,n,z = g.dynamic_link_prediction(data, None, zs, start=SKIP-1)
        
            t, f = model.score_fn(p,n,z)
            dscores = get_score(t, f)

            p,n,z = g.dynamic_new_link_prediction(data, None, zs, start=SKIP)
            if model.__class__ in uses_priors:
                z = zs 
            
            t, f = model.score_fn(p,n,z)
            nscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Dynamic LP:     %s
                    Dynamic New LP: %s
                ''' %
                (fmt_score(dscores), fmt_score(nscores))
            )

            return {
                'pred-auc': dscores[0],
                'pred-ap': dscores[1],
                'new-auc': nscores[0], 
                'new-ap': nscores[1]
            }


if __name__ == '__main__':
    data = vd.load_vgrnn('dblp')
    print(data.x.size(0))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        default='tgcn',
        help="Determines which model used from ['(T)GCN', '(R)GAE', '(V)GRNN', (P)TGCN]"
    )
    parser.add_argument(
        '-n', '--not-variational',
        action='store_false',
        help="Sets model to non-variational if flag used"
    )
    parser.add_argument(
        '-g', '--grnn',
        action='store_true',
        help='Uses Graph RNN if flag used'
    )
    parser.add_argument(
        '-s', '--static',
        action='store_false',
        help='Sets model to train on static link prediction'
    )
    parser.add_argument(
        '-l', '--sparse-loss',
        action='store_false',
        help='Uses the sparse loss function for VGRNN'
    )
    args = parser.parse_args()
    mtype = args.model.lower()
    outf = mtype + '.txt'

    for d in ['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d)
        
        if mtype == 'tgcn' or mtype == 't':
            if args.grnn:
                model = SerialTGCNGraphGRU(
                data.x.size(1), 32, 16        
            )
            else:    
                model = SerialTGCN(
                    data.x.size(1), 32, 16, 
                    variational=args.not_variational
                    #dense_loss=args.sparse_loss
                )
        elif mtype == 'rgae' or mtype == 'r':
            model = GAE_RNN(
                data.x.size(1), 32, 16,
                grnn=args.grnn, variational=args.not_variational
            )

        elif mtype == 'vgrnn' or mtype == 'v':
            model = VGRNN(
                data.x.size(1), 32, 16, pred=args.static,
                adj_loss=args.sparse_loss
            )

        elif mtype == 'ptgcn' or mtype == 'p':
            model = PriorSerialTGCN(
                data.x.size(1), 32, 16, pred=args.static
            )

        else: 
            raise Exception("Model must be one of ['TGCN', 'PTGCN', 'RGAE', 'VGRNN']")

        stats = [train(deepcopy(model), data, dynamic=args.static) for _ in range(NUM_TESTS)]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()