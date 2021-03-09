import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv 

from .serial_model import SerialTGCN

class VGCN_RetDistros(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VGCN_RetDistros, self).__init__()

        self.c1 = GCNConv(x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)

        self.mean = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.std = GCNConv(h_dim, z_dim, add_self_loops=True)

        self.soft = nn.Softplus()
        self.eps = 1e-10

    def forward(self, x, ei, ew=None):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        # x = self.drop(x)
        
        mean = self.mean(x, ei)
        if self.eval:
            return mean, torch.zeros((1)), torch.zeros((1))

        std = self.soft(self.std(x, ei))
        z = self._reparam(mean, std)

        return z, mean, std

    def _reparam(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

    def kld_loss(self, mean, std):
        return 0.5 * torch.sum(torch.exp(std) + mean**2 - 1. - std)

    '''
    Copied straight from the VGRNN code
    '''
    def kld_prior(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

'''
Attempts to use the previous h as a prior for 
the TGCN s.t. f(h_{t-1}) can predict the embeddings at time t
the same way VGRNN does. However, by using the TGCN model, the
time steps can all run through the GCN in parallel, allowing for
(maybe) faster run times if distributed


WARNING this model sucks hard. Don't use it
'''
class PriorSerialTGCN(SerialTGCN):
    def __init__(self, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                dynamic_feats=False, pred=True):
        super().__init__(
            x_dim, h_dim, z_dim, 
            gru_hidden_units=gru_hidden_units, 
            dynamic_feats=dynamic_feats, 
            variational=True
        )

        self.gcn = VGCN_RetDistros(x_dim, h_dim, h_dim)

        # Network to extract future predictions from prior hidden input
        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        self.pred = pred

    '''
    Iterates through list of xs, and eis passed in (if dynamic_feats is false
    assumes xs is a single 2d tensor that doesn't change through time)
    '''
    def forward(self, xs, eis, mask_fn, ews=None, start_idx=0):
        self.kld = torch.zeros((1))
        embeds, means, stds = self.encode(xs, eis, mask_fn, ews, start_idx)
        return self.recurrent(embeds, means, stds)

    '''
    Split proceses in two to make it easier to combine embeddings with 
    different masks (ie allow train set to influence test set embeds)
    '''
    def encode(self, xs, eis, mask_fn, ews=None, start_idx=0):
        embeds = []
        means = []
        stds = []
        
        for i in range(len(eis)):    
            ei = mask_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]

            z, mean, std = self.gcn(x,ei)
            means.append(mean)
            stds.append(std)
            embeds.append(z)

        return torch.stack(embeds), means, stds

    '''
    Generates Z embeddings and compares them to predictions from priors
    KLD field is back propped in the loss_fn call
    '''
    def recurrent(self, embeds, means, stds):
        z, h = self.gru(torch.tanh(embeds), include_h=True)

        prior = self.prior(h)
        pmean = self.prior_mean(prior)
        pstd = self.prior_std(prior)

        for i in range(h.size(0)-1):
            # Difference between predicted distro and actual
            self.kld += self.gcn.kld_prior(
                means[i+1], stds[i+1],  # actual
                pmean[i], pstd[i]       # predicted
            )

        self.kld = self.kld.true_divide(h.size(0)-1)

        if self.pred and self.eval:
            return pmean
        else:
            return z