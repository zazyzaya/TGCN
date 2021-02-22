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

'''
Splits edges into 85:5:10 train val test partition
(Following route of VGRNN paper)
'''
def edge_tvt_split(ei):
    ne = ei.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True 
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True 

    return masks[0], masks[1], masks[2]

'''
For the cyber data, all of the test set is the latter time
stamps. So only need train and val partitions
'''
def edge_tv_split(ei, v_size=0.05):
    ne = ei.size(1)
    val = int(ne*v_size)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True 

    return masks[0], masks[1]
