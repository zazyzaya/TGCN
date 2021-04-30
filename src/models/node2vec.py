import time 

import torch 
from torch.optim import SparseAdam
from torch_geometric.nn import Node2Vec

'''
Generates n2v embeddings for all edges in the model
'''
def embed(epochs, data, embedding_dim, walk_length, context_size,
            walks_per_node=1, p=1, q=1, num_negative_samples=1,
            lr=0.01):
            
    ei = torch.cat([data.eis[i] for i in range(data.T)], dim=1)
    model = Node2Vec(
        ei, embedding_dim, walk_length, context_size,
        walks_per_node, p, q, num_negative_samples, data.x.size(0),
        sparse=True
    )

    batch = torch.tensor(list(range(data.x.size(0))))
    opt = SparseAdam(model.parameters(), lr=lr)

    for e in range(epochs):
        s = time.time()
        opt.zero_grad()
        loss = model.loss(
            model.pos_sample(batch),
            model.neg_sample(batch)
        )

        loss.backward()
        opt.step()

        print("[%d] %0.04f  %0.4fs" % (e, loss.item(), time.time()-s))

    with torch.no_grad():
        x = model.forward()

    return x