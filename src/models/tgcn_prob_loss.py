import sys 
import torch 
from torch import nn 
from torch.autograd import Variable
from zayas_graph_modules.samplers import SampleMean

from .serial_model import SerialTGCN

class SoftmaxDetector(nn.Module):
    def __init__(self, in_feats, num_nodes):
        super().__init__()

        self.H = SampleMean()
        self.W = nn.Linear(in_feats, num_nodes)
        self.sm = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, ei):
        H = self.H(x, ei)
        probs = self.W(H)

        # Don't really need to return probs, just calculate loss
        # and return sampled embeds
        if self.training:
            return H, self.loss_fn(
                probs[ei[0]],
                ei[1]
            )

        # But it's sort of expensive to generate loss every time, 
        # so we can just return empty loss when eval        
        return H, torch.zeros((1))

    def score(self, H, src, dst):
        probs = self.sm(self.W(H))
        src_score = probs[src, dst]
        dst_score = probs[dst, src]

        return (src_score+dst_score)*0.5


class SoftmaxPredictor(SoftmaxDetector):
    '''
    Uses Zs from static LP to predict the next snapshot
    '''
    def forward(self, x, ei, next_ei):
        H = self.H(x, ei)
        probs = self.W(H)

        if self.training:
            return H, self.loss_fn(
                probs[next_ei[0]],
                next_ei[1]
            )

        return H, torch.zeros((1))

class ProbTGCN(SerialTGCN):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                use_w=True, lstm=False, neg_weight=0.5):
        super().__init__(
            x_dim, h_dim, z_dim, gru_hidden_units=gru_hidden_units, 
            dynamic_feats=False, dense_loss=False, 
            use_predictor=False, use_w=use_w, lstm=lstm, 
            neg_weight=neg_weight
        )

        # Assumes x_dim == num_nodes (which is true for all data we test
        self.anom_loss = None
        self.scorer = SoftmaxDetector(z_dim, x_dim)

    '''
    This works really well, but it's intractable for larger
    datasets, so omit for fairness in testing benchmarks
    def decode(self, src, dst, H, **kwarg):
        return self.scorer.score(H, src, dst)
    '''

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        # Calculate loss for static embeddings
        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]
      
            tot_loss += self.calc_loss(
                self.decode(t_src, t_dst, z),
                self.decode(f_src, f_dst, z)
            )   

        # Then add loss for dynamic PDFs
        return tot_loss.true_divide(T) + self.anom_loss

    def score_fn(self, ts, fs, H, **kwargs):
        t_scores, f_scores = [], []
        
        for i in range(H.size(0)):
            emb = H[i]
            t = ts[i]
            f = fs[i]

            t_scores.append(self.scorer.score(emb, t[0], t[1]))
            f_scores.append(self.scorer.score(emb, f[0], f[1]))

        return torch.cat(t_scores, dim=0), torch.cat(f_scores, dim=0)

    def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0, include_h=False, h_0=None):
        zs = super().forward(xs, eis, mask_fn, ew_fn, start_idx)

        hs, anom_loss = [], torch.zeros((1))
        for i in range(zs.size(0)):
            z = Variable(zs[i])
            ei = eis[i]

            h, loss = self.scorer(z, ei)
            hs.append(h)
            anom_loss += loss 

        hs = torch.stack(hs)
        self.anom_loss = anom_loss.true_divide(hs.size(0))
        return zs,hs

class HybridProbTGCN(SerialTGCN):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                use_w=True, lstm=False, neg_weight=0.5):
        super().__init__(
            x_dim, h_dim, z_dim, gru_hidden_units=gru_hidden_units, 
            dynamic_feats=False, dense_loss=False, 
            use_predictor=False, use_w=use_w, lstm=lstm, 
            neg_weight=neg_weight
        )
        # Assumes x_dim == num_nodes (which is true for all data we test
        self.anom_loss = None
        self.scorer = SoftmaxPredictor(z_dim, x_dim)


    def forward(self, xs, eis, mask_fn, ew_fn=None, start_idx=0, include_h=False, h_0=None):
        zs = super().forward(xs, eis, mask_fn, ew_fn, start_idx)

        hs, anom_loss = [], torch.zeros((1))
        for i in range(zs.size(0)-1):
            z = Variable(zs[i])
            ei = eis[i]
            next_ei = eis[i+1]

            h, loss = self.scorer(z, ei, next_ei)
            hs.append(h)
            anom_loss += loss 

        hs = torch.stack(hs)
        self.anom_loss = anom_loss.true_divide(hs.size(0))
        return zs,hs


    def score_fn(self, ts, fs, H, **kwargs):
        t_scores, f_scores = [], []
        
        for i in range(H.size(0)-1):
            emb = H[i]
            t = ts[i]
            f = fs[i]

            t_scores.append(self.scorer.score(emb, t[0], t[1]))
            f_scores.append(self.scorer.score(emb, f[0], f[1]))

        return torch.cat(t_scores, dim=0), torch.cat(f_scores, dim=0)

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1))
        T = len(ts)

        # Calculate loss for static embeddings
        for i in range(T):
            t_src, t_dst = ts[i]
            f_src, f_dst = fs[i]
            z = zs[i]
      
            tot_loss += self.calc_loss(
                self.decode(t_src, t_dst, z),
                self.decode(f_src, f_dst, z)
            )   

        # Then add loss for dynamic PDFs
        return tot_loss.true_divide(T) + self.anom_loss