import torch 

from torch import full, nn 
from torch.autograd import Variable
from torch_geometric.nn import GCNConv

from .gcn_gru import GraphGRU
from .loss_fns import full_adj_nll

class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GAE, self).__init__()

        self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.c1 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.sig = nn.Sigmoid()

    def forward(self, x, ei, ew=None):
        x = self.lin(x)
        # x = self.drop(x)
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        # x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        # x = self.drop(x)

        return x


class VGAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, embed_dim):
        super(VGAE, self).__init__()

        self.c1 = GCNConv(x_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.mean = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.std = GCNConv(hidden_dim, embed_dim, add_self_loops=True)

        self.soft = nn.Softplus()

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        
        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1))

        std = self.soft(self.std(x, ei))

        z = self._reparam(mean, std)
        kld = 0.5 * torch.sum(torch.exp(std) + mean**2 - 1. - std)

        return z, kld

    def _reparam(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)


class Recurrent(nn.Module):
    def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=1):
        super(Recurrent, self).__init__()

        self.gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=hidden_units, dropout=0.25
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_dim, out_dim)
        
        self.out_dim = out_dim 

    '''
    Expects (t, batch, feats) input
    Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
    '''
    def forward(self, xs, include_h=False):
        xs = self.drop(xs)
        xs, _ = self.gru(xs)
        
        if not include_h:
            return self.lin(xs)
        else:
            return self.lin(xs), xs


class SerialTGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                dynamic_feats=False, variational=False, dense_loss=False):
        super(SerialTGCN, self).__init__()

        self.dynamic_feats = dynamic_feats

        self.gcn = GAE(
            x_dim, embed_dim=h_dim, 
            hidden_dim=h_dim
        ) if not variational else VGAE(
            x_dim, h_dim, h_dim
        )

        self.gru = Recurrent(
            h_dim, out_dim=z_dim, 
            hidden_dim=h_dim, 
            hidden_units=gru_hidden_units
        ) if gru_hidden_units > 0 else None

        self.sig = nn.Sigmoid()
        self.kld = torch.zeros((1))
        self.variational = variational

        self.dense_loss=dense_loss
        msg = "dense" if self.dense_loss else 'sparse'
        print("Using %s loss" % msg)

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ews=None, start_idx=0):
        self.kld = torch.zeros((1))
        embeds = self.encode(xs, eis, mask_fn, ews, start_idx)

        return embeds \
            if type(self.gru) == type(None) \
            else self.gru(torch.tanh(embeds))

    '''
    Split proceses in two to make it easier to combine embeddings with 
    different masks (ie allow train set to influence test set embeds)
    '''
    def encode(self, xs, eis, mask_fn, ews=None, start_idx=0):
        embeds = []
        
        for i in range(len(eis)):    
            ei = mask_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]

            if self.variational: 
                z, kld = self.gcn(x,ei)
                self.kld += kld
            else:
                z = self.gcn(x,ei)

            embeds.append(z)

        self.kld = self.kld.true_divide(len(eis))
        return torch.stack(embeds)

    def recurrent(self, embeds):
        return self.gru(torch.tanh(embeds))


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
    Expects a list of true edges and false edges from each time
    step. Note: edge lists need not be the same length. Requires
    less preprocessing but doesn't utilize GPU/tensor ops as effectively
    as the batched fn  
    '''
    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            if not self.dense_loss:
                tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )   
            else:
                tot_loss = full_adj_nll(ts[i], z)

        return tot_loss.true_divide(T) + self.kld

    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL 
    '''
    def score_fn(self, ts, fs, zs):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        return tscores, fscores
            

    '''
    Returns inner product of src/dst nodes 
    '''
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return self.sig(dot)


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
        


class SerialTGCNGraphGRU(SerialTGCN):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1,
                dynamic_feats=False):
        
        super(SerialTGCNGraphGRU, self).__init__(
            x_dim, h_dim, z_dim,
            gru_hidden_units=gru_hidden_units,
            dynamic_feats=dynamic_feats
        )

        # Just changing the RNN mechanism 
        self.gru = GraphGRU(
            h_dim, z_dim,
            n_layers=gru_hidden_units
        ) if gru_hidden_units > 0 else None

    
    def forward(self, xs, eis, mask_fn, ews=None, start_idx=0):
        embeds = torch.tanh(self.encode(xs, eis, mask_fn, ews, start_idx))

        return embeds \
            if type(self.gru) == type(None) \
            else self.gru(embeds, eis, mask_fn)