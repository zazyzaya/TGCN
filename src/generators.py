from numpy.core.fromnumeric import partition
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
def link_prediction(data, partition_fn, zs, start=0, end=None, 
                    include_tr=True, batched=False, nratio=1):
    if batched:
        raise NotImplementedError("Sorry, batching is a TODO")

    end = end if end else data.T
    negs = []
    
    if partition_fn == None:
        partition_fn = lambda x : data.eis[x]

    for t in range(start, end):
        ei = tp = partition_fn(t)
        
        # Also avoid edges from training set (assuming this is val or test 
        # calling the function)
        if include_tr:
            ei = torch.cat([ei, data.tr(t)], dim=1)

        neg = fast_negative_sampling(ei, int(tp.size(1)*nratio), data.num_nodes)
        negs.append(neg)

    return [partition_fn(i) for i in range(start, end)], negs, zs 

'''
Using embeddings from timestep t, predict links in timestep t+1
same as link prediction, just offset edge lists and embeddings by -1
'''
def dynamic_link_prediction(data, partition_fn, zs, start=0, end=None, 
                            include_tr=True, batched=False, nratio=1):
    # Uses every edge in the next snap shot, so no partition fn needed
    p, n, z = link_prediction(
        data, partition_fn, zs, start, end, 
        include_tr, batched, nratio
    )

    p = p[1:]
    n = n[1:]
    z = z[:-1]

    return p, n, z
    
'''
Predict links that weren't present in prev batch appearing in next batch 
(Compute heavy. May want to precalculate this/only run on test set)
'''
def dynamic_new_link_prediction(data, partition_fn, zs, start=0, end=None, 
                                include_tr=True, batched=False):
    if batched:
        raise NotImplementedError("Sorry, batching is a TODO")

    p, n = [], []
    b = None

    if partition_fn == None:
        partition_fn = lambda x : data.eis[x]

    end = end if end else data.T

    for i in range(start, end):
        # Use full adj matrix for new link pred
        ei = partition_fn(i)

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
                ei, p[-1].size(1), data.num_nodes
            )
        )

    return p, n, zs[:-1]