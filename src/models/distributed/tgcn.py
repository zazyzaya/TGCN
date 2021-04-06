from copy import deepcopy

import torch 
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP

from ..serial_model import GAE, VGAE, SerialTGCN
from .utils import _remote_method, _remote_method_async, _param_rrefs

'''
Extended so data can live in these objects to minimize 
needless communication 

TODO 
'''
class R_VGAE(VGAE):
    def __init__(self, x_dim, hidden_dim, embed_dim, data):
        super().__init__(x_dim, hidden_dim, embed_dim)
        self.data = data 

    def forward(self, mask):
        zs = []
        for i in range(self.data.T):
            x = self.data.x 
            ei = self.data.masked(i, mask)

'''
Called by worker processes to initialize their models
The RRefs to these models are then passed to the TGCN
'''
def get_remote_gae(x_dim, h_dim, variational=True):
    m = 'VGAE' if variational else 'GAE'

    print("Building " + m + " on " + rpc.get_worker_info().name)
    if variational:
        model = VGAE(
            x_dim, h_dim, h_dim
        )
    else:
        model = GAE(
            x_dim, embed_dim=h_dim, hidden_dim=h_dim
        )

    model.train()
    return DDP(model)    


class TGCN(SerialTGCN):
    def __init__(self, remote_rrefs, x_dim, h_dim, z_dim, gru_hidden_units=1, 
                dynamic_feats=False, variational=True):
        
        super(TGCN, self).__init__(
            x_dim, h_dim, z_dim, gru_hidden_units=gru_hidden_units, 
            dynamic_feats=dynamic_feats, variational=variational, 
            dense_loss=False, use_predictor=False, use_graph_gru=False
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
    
        embeds = [f.wait() for f in embed_futs]
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
        
        return params

    '''
    Makes a copy of the current state dict as well as 
    the distributed GCN state dict (just worker 0)
    '''
    def save_states(self):
        gcn = _remote_method(
            DDP.state_dict, self.gcns[0]
        )

        return gcn, deepcopy(self.state_dict())

    '''
    Given the state dict for one GCN and the RNN load them
    into the dist and local models
    '''
    def load_states(self, gcn_state_dict, rnn_state_dict):
        self.load_state_dict(rnn_state_dict)
        
        jobs = []
        for rref in self.gcns:
            jobs.append(
                _remote_method_async(
                    DDP.load_state_dict, rref, 
                    gcn_state_dict
                )
            )

        [j.wait() for j in jobs]