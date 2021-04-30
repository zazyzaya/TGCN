import torch 

from torch import nn 
from torch_geometric.nn import GCNConv

from .gcn_gru import GraphGRU
from .loss_fns import full_adj_nll

class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GAE, self).__init__()

        #self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
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

        return x


class Recurrent(nn.Module):
    def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=1):
        super(Recurrent, self).__init__()

        self.gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=hidden_units#, dropout=0.25
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_dim, out_dim)#, bias=False)
        
        self.out_dim = out_dim 

    '''
    Expects (t, batch, feats) input
    Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
    '''
    def forward(self, xs, h_0, include_h=False):
        xs = self.drop(xs)
        if type(h_0) != type(None):
            xs, h = self.gru(xs, h_0)
        else:
            xs, h = self.gru(xs)
        
        #xs = self.drop(xs)
        if not include_h:
            return self.lin(xs)
        else:
            return self.lin(xs), h


class SerialTGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                dynamic_feats=False, dense_loss=False,
                use_predictor=False, use_w=True,
                neg_weight=0.5):
        super(SerialTGCN, self).__init__()

        self.weightless = not use_w
        self.kld_weight = 0
        self.dynamic_feats = dynamic_feats
        self.neg_weight = neg_weight
        self.cutoff = None

        self.gcn = GAE(
            x_dim, embed_dim=h_dim, 
            hidden_dim=h_dim
        ) 

        self.gru = Recurrent(
            h_dim, out_dim=z_dim, 
            hidden_dim=h_dim, 
            hidden_units=gru_hidden_units
        ) if gru_hidden_units > 0 else None

        self.use_predictor = use_predictor
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        ) if use_predictor else None

        self.sig = nn.Sigmoid()

        self.dense_loss=dense_loss
        msg = "dense" if self.dense_loss else 'sparse'
        print("Using %s loss" % msg)

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0, 
                include_h=False, h_0=None):
        embeds = self.encode(xs, eis, mask_fn, ew_fn, start_idx)

        if type(self.gru) == type(None):
            if not include_h:
                return embeds
            else:
                return embeds, None
        else:
            return self.gru(torch.tanh(embeds), h_0, include_h=include_h)
        

    '''
    Split proceses in two to make it easier to combine embeddings with 
    different masks (ie allow train set to influence test set embeds)
    '''
    def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
        embeds = []
        
        for i in range(len(eis)):    
            ei = mask_fn(start_idx + i)
            ew = None if not ew_fn or self.weightless else ew_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]

            z = self.gcn(x,ei,ew)
            #print("ei: %s  ew: %s" % (str(ei.size()), str(ew.size())))
            embeds.append(z)

        return torch.stack(embeds)

    def fast_encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
        num_nodes = xs.size(0) if not self.dynamic_feats else xs[0].size(0)

        T = len(eis)
        eis = torch.cat(
            [
                mask_fn(start_idx+i) + (i*xs.size(0))
                for i in range(T)
            ], dim=1
        )
        print('eis:')
        print(eis.size())
        
        if not ew_fn or self.weightless:
            ews = None
        else:
            ews = torch.cat(
                [
                    ew_fn(start_idx+i)
                    for i in range(T)
                ], dim=0
            )
            print('\news:')
            print(ews.size())

        if not self.dynamic_feats:
            xs = xs.repeat(T, 1)
            print("\nXs:")
            print(xs.size())

        zs = self.gcn(xs, eis, ews)

        return zs.reshape(T, num_nodes, zs.size(1))

    def recurrent(self, embeds):
        x = self.gru(
            torch.tanh(embeds)
        )

        if self.graph_gru:
            x = self.drop(x)
            return self.gru_lin(x)
        return x


    '''
    Inner product given edge list and embeddings at time t
    '''
    def decode(self, src, dst, z, as_probs=False):
        if self.use_predictor:
            return self.predictor(
                z[src] * z[dst]
            )
        
        dot = (z[src] * z[dst]).sum(dim=1)
        logits = self.sig(dot)

        if as_probs:
            return self.__logits_to_probs(logits)
        return logits


    '''
    Given confidence scores of true samples and false samples, return
    neg log likelihood 
    '''
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (1-self.neg_weight) * pos_loss + self.neg_weight * neg_loss


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

        return tot_loss.true_divide(T)

    '''
    Get scores for true/false embeddings to find ROC/AP scores.
    Essentially the same as loss_fn but with no NLL 

    Returns logits unless as_probs is True
    '''
    def score_fn(self, ts, fs, zs, as_probs=False):
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

        if as_probs:
            tscores=self.__logits_to_probs(tscores)
            fscores=self.__logits_to_probs(fscores)

        return tscores, fscores


    '''
    Converts from log odds (what the encode method outputs) to probabilities
    '''
    def __logits_to_probs(self, logits):
        odds = torch.exp(logits)
        probs = odds.true_divide(1+odds)
        return probs