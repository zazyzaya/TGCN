import torch 

from torch import nn 
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GCN, self).__init__()

        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.sig = nn.Sigmoid()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.drop(x)

        return self.sig(x)

class Recurrent(nn.Module):
    def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=16):
        super(Recurrent, self).__init__()

        self.gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=hidden_units
        )

        self.lin = nn.Linear(hidden_dim, out_dim)
        self.sig = nn.Sigmoid()
        
        self.out_dim = out_dim 
            

    '''
    Expects (t, batch, feats) input
    Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
    '''
    def forward(self, xs):
        xs = self.gru(xs)
        return self.lin(xs)

    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, z, src, dst):
        dot = (z[src] * z[dst]).sum(dim=1)
        return self.sig(dot) 

    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood 
    '''
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss

    '''
    Expects stacks of 3D tensors s.t. ts[0] contains each src node
    the rows of ts[0] are different time steps, cols are different nodes.
    This requires some preprocessing and probably minibatching to make 
    sure edge lists are all the same length, but it's a lot faster
    '''
    def loss_fn_batched(self, ts, fs, zs):
        t_scores = self.batch_decode(zs, ts[0], ts[1])
        f_scores = self.batch_decode(zs, fs[0], fs[1])

        return self.calc_loss(t_scores, f_scores)

    '''
    Does 3D gather before performing the same function as self.decode
    '''
    def batch_decode(self, z, src, dst):
        zsrc = self.__3d_gather(z, src)
        zdst = self.__3d_gather(z, dst)

        dot = (zsrc * zdst).sum(dim=1)
        return self.sig(dot)

    '''
    Helper function for batched loss. Uses gather to grab rows from 
    t assuming each index in idx represents a row of features from 
    0th dimension corresponding to idx in index's row
    '''
    def __3d_gather(self, t, idx):
        # First grab the full rows of t (hence the repeat method)
        ret = torch.gather(t, 1, idx.unsqueeze(-1).repeat((1,1,self.out_dim)))
        
        # Reshape to 2D 
        ret = ret.view(idx.size(0)*idx.size(1), self.out_dim)