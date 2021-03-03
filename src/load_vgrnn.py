import os 
import pickle

import torch
from torch_geometric.utils import dense_to_sparse 
from torch_geometric.data import Data

from utils import edge_tvt_split

class TData(Data):
    def __init__(self, **kwargs):
        super(TData, self).__init__(**kwargs)

        # Getter methods so I don't have to write this every time
        self.va = lambda t : self.eis[t][:, self.masks[t][1]]
        self.te = lambda t : self.eis[t][:, self.masks[t][2]]

        # Only the last 3 time stamps are masked
        self.tr = lambda t : self.eis[t][:, self.masks[t][0]] #if t >= self.T-3 else self.eis[t]
        self.all = lambda t : self.eis[t]

'''
For loading datasets from the VRGNN repo (none have features)
'''
def load_vgrnn(dataset):
    datasets = ['fb', 'dblp', 'enron10']
    assert dataset in datasets, \
        "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

    adj = os.path.join('/mnt/raid0_24TB/isaiah/code/TGCN/src/data', dataset, 'adj_orig_dense_list.pickle')
    with open(adj, 'rb') as f:
        fbytes = f.read() 

    dense_adj_list = pickle.loads(fbytes, encoding='bytes')
    num_nodes = dense_adj_list[0].size(0)
    
    eis = []
    splits = []

    for adj in dense_adj_list:
        # Remove self loops
        for i in range(adj.size(0)):
            adj[i,i] = 0

        ei = dense_to_sparse(adj)[0]
        
        eis.append(ei)
        splits.append(edge_tvt_split(ei))


    data = TData(
        x=torch.eye(num_nodes),
        eis=eis,
        masks=splits,
        num_nodes=num_nodes,
        T=len(eis)
    )    

    return data 