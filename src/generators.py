import torch 
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from utils import fast_negative_sampling


# # # # # # # # # # # # # # # # # # # # # # # # # #
#          Generators for data splits for         #
#           training/testing/validation           #
#       Each returns tuple (pos, neg, embeds)     #
#      of pos and negative edge lists, and the    #
#      embeddings they reference, respectively    #
# # # # # # # # # # # # # # # # # # # # # # # # # #

'''
Simplest one. Just returns eis and random negative sample
for each time step AT each timestep
'''
def link_prediction(data, partition_fn, zs, trange=None, include_tr=True, batched=False):
    if batched:
        raise NotImplementedError("Sorry, batching is a TODO")

    trange= range(data.T) if trange==None else trange
    negs = []

    for t in trange:
        ei = partition_fn(t)
        
        # Also avoid edges from training set (assuming this is val or test 
        # calling the function)
        if include_tr:
            ei = torch.cat([ei, data.tr(t)], dim=1)

        neg = fast_negative_sampling(ei, ei.size(1), data.num_nodes)
        negs.append(neg)

    return [data.eis[i] for i in trange], negs, zs 

'''
Using embeddings from timestep t, predict links in timestep t+1
same as link prediction, just offset edge lists and embeddings by -1
'''
def dynamic_link_prediction(data, partition_fn, zs, trange=None, include_tr=True, batched=False):
    p, n, z = link_prediction(data, partition_fn, zs, trange, include_tr, batched)

    p = p[1:]
    n = n[1:]
    z = z[:-1]

    return p, n, z
    
'''
Predict links that weren't present in prev batch appearing in next batch 
(Compute heavy. May want to precalculate this/only run on test set)
'''
def dynamic_new_link_prediction(data, partition_fn, zs, trange=None, include_tr=True, batched=False):
    if batched:
        raise NotImplementedError("Sorry, batching is a TODO")

    p, n = [], []
    b = None
    trange= range(data.T) if trange==None else trange

    for i in trange:
        ei = partition_fn(i)

        if include_tr:
            ei = torch.cat([ei, data.tr(i)], dim=1)

        a = b
        b = to_dense_adj(ei, max_num_nodes=data.num_nodes)[0].bool()

        if type(a) == type(None):
            continue 

        # Generates new links in next time step
        new_links = (~a).logical_and(a.logical_or(b))
        new_links, _ = dense_to_sparse(new_links)

        p.append(new_links)
        n.append(
            fast_negative_sampling(
                ei, ei.size(1), data.num_nodes
            )
        )

    return p, n, zs[:-1]