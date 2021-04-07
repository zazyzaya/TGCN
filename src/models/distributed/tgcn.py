from copy import deepcopy

import torch 
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP

from ..serial_model import GAE, VGAE, SerialTGCN
from .utils import _remote_method, _remote_method_async, _param_rrefs

'''
Extended so data can live in these objects to minimize 
needless communication 
'''
class R_GAE(GAE):
    def __init__(self, h_dim, loader, load_args):
        print(rpc.get_worker_info().name + ": Loading ts from %d to %d" % (load_args['start'], load_args['end']))

        data = loader(**load_args) 
        super(R_GAE, self).__init__(data.x.size(1), h_dim, h_dim)
        
        self.data = data 
        self.x_dim = data.x.size(1)

    def forward(self, mask):
        zs = []
        for i in range(self.data.T):
            x = self.data.x 
            ei, ew = self.data.masked(i, mask)
            zs.append(
                super(R_GAE, self).forward(
                    x, ei, ew=ew
                )
            )

        return torch.stack(zs)

'''
Allows you to call methods other than forward from the 
RPC remote method
'''
class MyDDP(DDP):
    '''
    Used to generate negative edges by master
    (Unaccessable with DDP... TODO)
    '''
    def get_eis(self, mask_enum):
        return [
            self.module.data.masked(i, mask_enum)[0]
            for i in range(self.module.data.T)
        ]

    def get_x_dim(self):
        return self.module.x_dim

'''
Called by worker processes to initialize their models
The RRefs to these models are then passed to the TGCN
'''
def get_remote_gae(h_dim, loader, load_args): 
    model = R_GAE(
        h_dim, loader, load_args
    )

    return MyDDP(model)    


class TGCN(SerialTGCN):
    def __init__(self, remote_rrefs, h_dim, z_dim, 
                gru_hidden_units=1, dynamic_feats=False):
        
        # X_dim doesn't matter, GAEs take care of it
        super(TGCN, self).__init__(
            1, h_dim, z_dim, gru_hidden_units=gru_hidden_units, 
            dynamic_feats=dynamic_feats, variational=False, 
            dense_loss=False, use_predictor=False, use_graph_gru=False
        )

        self.dynamic_feats = dynamic_feats
        
        # Only difference is that the GCN is now many GCNs sharing params
        # across several computers/processors
        del self.gcn 
        self.gcns = remote_rrefs
        self.num_workers = len(self.gcns)


    '''
    Only need to tell workers which mask to use on the data, nothing else
    is needed. 
    '''
    def forward(self, mask_enum, include_h=False, h_0=None):
        zs = self.encode(mask_enum)
        return self.gru(torch.tanh(zs), h_0, include_h=include_h)

    '''
    Tell each remote GCN to encode their data. Data lives there to minimise 
    net traffic 
    '''
    def encode(self, mask_enum):
        embed_futs = []
        
        for i in range(self.num_workers):    
            embed_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.gcns[i % self.num_workers],
                    mask_enum
                )
            )
    
        embeds = [f.wait() for f in embed_futs]
        return torch.cat(embeds, dim=0)

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