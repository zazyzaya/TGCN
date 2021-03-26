import torch 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP

from ..serial_model import GAE, VGAE, SerialTGCN
from .utils import _remote_method, _remote_method_async, _param_rrefs

'''
Called by worker processes to initialize their models
The RRefs to these models are then passed to the TGCN
'''
def get_remote_gae(x_dim, h_dim, variational=True):
    if variational:
        model = VGAE(
            x_dim, h_dim, h_dim
        )
    else:
        model = GAE(
            x_dim, embed_dim=h_dim, hidden_dim=h_dim
        )

    return DDP(model)


class TGCN(SerialTGCN):
    def __init__(self, remote_rrefs, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                dynamic_feats=False, variational=True, dense_loss=False,
                use_predictor=False, use_graph_gru=False):
        
        super(TGCN, self).__init__(
            x_dim, h_dim, z_dim, gru_hidden_units, 
            dynamic_feats, variational, dense_loss,
            use_predictor, use_graph_gru
        )

        self.dynamic_feats = dynamic_feats
        
        # Only difference is that the GCN is now many GCNs sharing params
        # across several computers/processors
        del self.gcn 
        self.gcns = remote_rrefs
        self.num_workers = len(self.gcns)


    '''
    Same as parent, but now we send each time step to a different worker
    
    TODO in the future, have the tensors preloaded on the worker machines
    to cut down communication costs
    '''
    def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
        embed_futs = []
        
        for i in range(len(eis)):    
            ei = mask_fn(start_idx + i)
            ew = None if not ew_fn else ew_fn(start_idx + i)
            x = xs if not self.dynamic_feats else xs[start_idx + i]

            embed_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.gcns[i % self.num_workers],
                    x, ei, ew=ew
                )
            )

        embeds = []
        if self.variational:
            for f in embed_futs:
                e, kld = f.wait() 
                embeds.append(e)
                self.kld += kld 
        else:
            embeds = [f.wait() for f in embed_futs]

        self.kld = self.kld.true_divide(len(eis))
        return torch.stack(embeds)

    
    '''
    Distributed optimizer needs RRefs to params rather than the literal
    locations of them that you'd get with self.parameters() 
    '''
    def parameter_rrefs(self):
        params = []
        for rref in self.gcns: 
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )
        
        params.extend(_param_rrefs(self.gru))
        params.extend(_param_rrefs(self.sig))