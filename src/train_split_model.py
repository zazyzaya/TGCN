from copy import deepcopy

import pandas as pd
import torch 
from torch.optim import Adam 

from generators import dynamic_new_link_prediction
from loaders.load_vgrnn import load_vgrnn, TData
from models.static_dynamic import *
from utils import get_score, fast_negative_sampling

torch.set_num_threads(4)
TR_PARAMS = {
    'lr': 0.01,
    'epochs': 1500,
    'patience': 50,
    'neg_samples': 10.0
}

def get_sample(data, enum, nsize=TR_PARAMS['neg_samples']):
    ps = [
            data.ei_masked(enum, i)
            for i in range(data.T)
        ]

    ns = [
        fast_negative_sampling(
            ps[i],
            int(ps[i].size(1) * nsize),
            data.num_nodes, 
        )
        for i in range(data.T)
    ]

    return ps, ns

def train(model, data):
    opt = Adam(model.parameters(), lr=TR_PARAMS['lr'])
    val_best = (0, None)
    no_change = 0 

    for e in range(TR_PARAMS['epochs']):
        model.train()
        opt.zero_grad()
        zs, preds = model.forward(data, TData.TR)
        ps, ns = get_sample(data, TData.TR)

        loss = model.loss_fn(ps, ns, zs, preds)
        print("[%d] Loss: %0.6f" % (e, loss.item()), end='')

        loss.backward()
        opt.step()
        
        with torch.no_grad():
            zs, preds = model.forward(data, TData.VA)
            ps, ns = get_sample(data, TData.VA)

            sscore, dscore = model.score(ps, ns, zs, preds)
            #loss = model.loss_fn(ps, ns, zs, preds).item()
            sscore = get_score(*sscore)
            dscore = get_score(*dscore)

            score = sum(sscore + dscore) / 4.0
            star = '*' if score > val_best[0] else ''

            #print('''\tEval loss: %0.6f%s
            #Static:
            #    AUC: %0.6f\tAP: %0.6f
            #Dynamic Eval:
            #    AUC: %0.6f\tAP: %0.6f
            #
            #''' % (loss, star, sscore[0], sscore[1], dscore[0], dscore[1]))
            print("\tEval score: %0.6f%s" % (score, star))

            if score < val_best[0]:
                no_change += 1
                if no_change > TR_PARAMS['patience']:
                    print("Early stopping!")
                    break 

            else:
                no_change = 0
                val_best = (score, deepcopy(model))

    
    return val_best[1]


def test(model, data, te_starts=3):
    model.eval()
    with torch.no_grad():
        zs, preds = model(data, TData.TR)

    ps, ns = get_sample(data, TData.TE, nsize=1)
    sscore = model.static.score(
        ps[-te_starts:], ns[-te_starts:], zs[-te_starts:]
    )

    dscore = model.dynamic.score(
        ps[-te_starts-1:], ns[-te_starts-1:], preds[-te_starts-1:]
    )

    dnp, dnn, _ = dynamic_new_link_prediction(data, data.te, preds)
    dnscore = model.dynamic.score(
        dnp[-te_starts-1:], dnn[-te_starts-1:], preds[-te_starts-1:]
    )
    
    sscore = get_score(*sscore)
    dscore = get_score(*dscore)
    dnscore = get_score(*dnscore)

    print('''Scores:
        Static AUC: %0.6f\tAP: %0.6f
        Dynamic AUC: %0.6f\tAP: %0.6f
        Dyn New AUC: %0.6f\tAP: %0.6f
    ''' % (
        sscore[0], sscore[1], 
        dscore[0], dscore[1],
        dnscore[0], dnscore[1]
    ))

    return {
        'static_auc': sscore[0],
        'static_ap': sscore[1],
        'dyn_auc': dscore[0],
        'dyn_ap': dscore[1],
        'new_auc': dnscore[0],
        'new_ap': dnscore[1]
    }


def main():
    for dataset in ['fb', 'enron10', 'dblp']:
        stats = []
        for _ in range(5):
            data_tr = load_vgrnn(dataset)
            data_tr.T -= 3
            data_tr.eis = data_tr.eis[:-3]

            model = GAE_LSTM(data_tr.x_dim, 32, 16)
            model = train(model, data_tr)

            data_te = load_vgrnn(dataset)
            stats.append(test(model, data_te))

        df = pd.DataFrame(stats)
        
        print(df.mean()*100)
        print(df.sem()*100)

        f = open('split_tests.txt', 'a')
        f.write(dataset + '\n')
        f.write(str(df.mean()*100) + '\n')
        f.write(str(df.sem()*100) + '\n\n')
        f.write(str(df) + '\n\n')
        f.close()

main()