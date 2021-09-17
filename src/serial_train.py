from copy import deepcopy
import argparse

import pandas as pd
import torch 
from torch.optim import Adam

import generators as g
import loaders.load_vgrnn as vd
from models.serial_model import SerialTGCN
from models.vgrnn_like import GAE_RNN, VGRNN
from models.tgcn_with_prior import PriorSerialTGCN
from models.tgcn_prob_loss import ProbTGCN
from utils import get_score

torch.set_num_threads(8)

uses_priors = [VGRNN, PriorSerialTGCN]

NUM_TESTS = 5

PATIENCE = 100
MAX_DECREASE = 2

KL_ANNEALING = 100
KL_WEIGHT = 0.0001

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
        zs = model(data.x, data.eis, data.tr)[:end_tr]

        if not dynamic or model.__class__ in uses_priors:
            p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, nratio=nratio)
            
        else:
            p,n,z = g.dynamic_link_prediction(data, data.tr, zs, include_tr=False, nratio=nratio)      
        

        loss = model.loss_fn(p,n,z)
        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 
        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr)[:end_tr]

            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[0] + sscores[1]

            else:
                # VGRNN is providing priors, which are built from the previous timestep
                # already, thus there is no need to shift the selected ei's as the 
                # dynamic functions do 
                if model.__class__ in uses_priors:
                    zs = zs[1:]
                    dp,dn,dz = g.link_prediction(data, data.va, zs)
                else:
                    dp,dn,dz = g.dynamic_link_prediction(data, data.va, zs)
                
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, data.va, zs)
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
                    print("Early stopping...\n")
                    break


    model = best[1]
    with torch.no_grad():
        model.eval()
        zs = model(data.x, data.eis, data.tr)[end_tr-1:]

        if not dynamic:
            zs = zs[1:]
            p,n,z = g.link_prediction(data, data.te, zs, start=end_tr)
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
            if model.__class__ in uses_priors:
                zs = zs[1:]
                p,n,z = g.link_prediction(data, data.te, zs, start=end_tr)
            else:                
                p,n,z = g.dynamic_link_prediction(data, data.te, zs, start=end_tr-1)
                print(len(p))
                print(z.size(0))

            t, f = model.score_fn(p,n,z)
            dscores = get_score(t, f)

            p,n,z = g.dynamic_new_link_prediction(data, data.te, zs, start=end_tr-1)
            if model.__class__ in uses_priors:
                z = zs 

            print(z.size(0))
            
            t, f = model.score_fn(p,n,z)
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
    #data = vd.load_vgrnn('dblp')
    #print(data.x.size(0))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        default='tgcn',
        type=str.lower,
        help="Determines which model used from ['(T)GCN', '(V)GRNN', (P)TGCN, (G)CN, (U)GAED]"
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
    parser.add_argument(
        '--lstm',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01
    )

    args = parser.parse_args()
    mtype = args.model[0].lower()
    outf = mtype + '.txt' 

    for d in ['enron10', 'fb', 'dblp']:
        data = vd.load_vgrnn(d)
        
        if mtype == 'tgcn' or mtype == 't':   
            model = SerialTGCN(
                data.x.size(1), 32, 16, lstm=True
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

        elif mtype == 'gcn' or mtype == 'g':
            model = SerialTGCN(
                data.x.size(1), 32, 16, 
                gru_hidden_units=0
            )

        elif mtype == 'ugaed' or mtype == 'u':
            model = ProbTGCN(
                data.x.size(1), 32, 16
            )

        else: 
            raise Exception("Model must be one of ['TGCN', 'PTGCN', 'RGAE', 'VGRNN']")

        stats = [train(deepcopy(model), data, dynamic=args.static, lr=args.lr) for _ in range(NUM_TESTS)]

        df = pd.DataFrame(stats)
        print(df.mean()*100)
        print(df.sem()*100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('LR: %0.4f\n' % args.lr)
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.close()