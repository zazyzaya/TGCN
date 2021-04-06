import torch 
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

'''
Returns AUC and AP scores given true and false scores
'''
def get_score(pscore, nscore):
    ntp = pscore.size(0)
    ntn = nscore.size(0)

    score = torch.cat([pscore, nscore]).numpy()
    labels = np.zeros(ntp + ntn, dtype=np.long)
    labels[:ntp] = 1

    ap = average_precision_score(labels, score)
    auc = roc_auc_score(labels, score)

    return [auc, ap]

'''
Calculates true positive rate and false positive rate given 
the rank of this anomalous edge, 
the number of anoms ranked higher than this one
the total number of edges
the total number of anomalies
'''
def tpr_fpr(rank, n, total, tot_anom):
    # TPR is easy
    tpr = n/rank

    # FPR is trickier 
    fp = rank-n
    tn = total-rank-tot_anom
    fpr = fp / (fp+tn)

    return "TPR: %0.4f, FPR: %0.4f" % (tpr*100, fpr*100)

'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin 
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, num_nodes, oversample=1.25):    
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample)))
        neg_hash = el_hash(maybe_neg)
        
        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )

    # May have gotten some extras
    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()