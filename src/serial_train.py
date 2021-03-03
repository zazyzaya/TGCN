from copy import deepcopy
import argparse
from re import M

import torch 
from torch.optim import Adam

import generators as g
import load_vgrnn as vd
import load_cyber as cd 
from models.serial_model import SerialTGCN, SerialTGCNGraphGRU
from models.vgrnn_like import GAE_RNN, VGRNN
from utils import get_score

torch.set_num_threads(16)

LR = 0.01
PATIENCE = 50

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])

def train(model, data, epochs=1500, dynamic=False):
    # Test/Val on last 3 time steps
    SKIP = data.T-3

    opt = Adam(model.parameters(), lr=LR)

    best = (0, None)
    no_improvement = 0
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        # Get embedding
        if dynamic:
            zs = model(data.x, data.eis[:SKIP], data.all)
        else:
            zs = model(data.x, data.eis[:SKIP], data.tr)
        
        # Calculate static or dynamic loss. Note that if training dynamic
        # the test set is the final 3 timesteps, rather than masked out edges
        # in timesteps before. Thus, all edges in timeslices t<T-SKIP are 
        # viewed by the model
        if dynamic:
            p,n,z = g.link_prediction(data, None, zs, include_tr=False, end=SKIP)
        else:
            p,n,z = g.link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)

        if not dynamic or model.__class__ == VGRNN:
            loss = model.loss_fn(p,n,z)
        
        # Other models can use more data
        else:
            dp,dn,dz = g.dynamic_link_prediction(data, data.tr, zs, include_tr=False, end=SKIP)
            loss = model.loss_fn(p+dp, n+dn, torch.cat([z, dz], dim=0))
            

        loss.backward()
        opt.step()

        # Done by VGRNN to improve convergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item() 

        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr)[SKIP:]
        
            if not dynamic:
                p,n,z = g.link_prediction(data, data.va, zs, start=SKIP)
                st, sf = model.score_fn(p,n,z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores) )
                )

                avg = sscores[1]

            else:
                dp,dn,dz = g.dynamic_link_prediction(data, None, zs, start=SKIP)
                dt, df = model.score_fn(dp,dn,dz)
                dscores = get_score(dt, df)

                dp,dn,dz = g.dynamic_new_link_prediction(data, None, zs, start=SKIP)
                dt, df = model.score_fn(dp,dn,dz)
                dnscores = get_score(dt, df)

                print(
                    '[%d] Loss: %0.4f  \n\tDet %s  \n\tNew %s' %
                    (e, trloss, fmt_score(dscores), fmt_score(dnscores) )
                )

                avg = (
                    dscores[1] +
                    dnscores[1]
                )
            
            if avg > best[0]:
                best = (avg, deepcopy(model))
                no_improvement = 0
            else:
                # Though it's not reflected in the code, the authors for VGRNN imply in the
                # supplimental material that after 500 epochs, early stopping may kick in 
                if e > 500:
                    no_improvement += 1
                if no_improvement > PATIENCE:
                    print("Early stopping...\n")
                    break


    model = best[1]
    with torch.no_grad():
        model.eval()
        zs = model(data.x, data.eis[SKIP:], data.tr, start_idx=SKIP)

        if not dynamic:
            p,n,z = g.link_prediction(data, data.te, zs, start=SKIP)
            t, f = model.score_fn(p,n,z)
            sscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Static LP:  %s
                '''
            % fmt_score(sscores))

            return sscores

        else:
            p,n,z = g.dynamic_link_prediction(data, None, zs, start=SKIP)
            t, f = model.score_fn(p,n,z)
            dscores = get_score(t, f)

            p,n,z = g.dynamic_new_link_prediction(data, None, zs, start=SKIP)
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

            return {'pred': dscores, 'new': nscores}


if __name__ == '__main__':
    data = vd.load_vgrnn('dblp')
    print(data.x.size(0))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model',
        default='tgcn',
        help="Determines which model used from ['(T)GCN', '(R)GAE', '(V)GRNN']"
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
    args = parser.parse_args()

    mtype = args.model.lower()
    if mtype == 'tgcn' or mtype == 't':
        if args.grnn:
            model = SerialTGCNGraphGRU(
            data.x.size(1),
            gcn_out_dim=32,
            gru_embed_dim=16,
            gru_hidden_units=1
        )
        else:    
            model = SerialTGCN(
                data.x.size(1),
                gcn_out_dim=32,
                gru_embed_dim=16,
                gru_hidden_units=1
            )
    elif mtype == 'rgae' or mtype == 'r':
        model = GAE_RNN(
            data.x.size(1),
            16, 32, grnn=args.grnn, variational=args.not_variational
        )

    elif mtype == 'vgrnn' or mtype == 'v':
        model = VGRNN(
            data.x.size(1), 16, 32
        )

    else: 
        raise Exception("Model must be one of ['TGCN', 'RGAE', 'VGRNN']")

    print(model.__class__)
    train(model, data, dynamic=args.static)