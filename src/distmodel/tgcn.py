from copy import deepcopy

import torch 
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP

from . import generators as g
from models.serial_model import GAE, SerialTGCN
from .utils import _remote_method, _remote_method_async, _param_rrefs

'''
Extended so data can live in these objects to minimize 
needless communication 
'''
class R_GAE(GAE):
    def __init__(self, h_dim, loader, load_args):
        print(rpc.get_worker_info().name + ": Loading ts from %d to %d" % (load_args['start'], load_args['end']))

        jobs = load_args.pop('jobs')
        data = loader(jobs, **load_args) 
        super(R_GAE, self).__init__(data.x.size(1), h_dim, h_dim)
        
        self.data = data 
        self.x_dim = data.x.size(1)

    def __forward(self, mask):
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
    Wrapper function for __forward, makes a little
    easier to read
    '''
    def forward(self, mask, no_grad):
        if no_grad:
            with torch.no_grad():
                return self.__forward(mask)
        
        return self.__forward(mask)

'''
Allows you to call methods other than forward from the 
RPC remote method
'''
class GAE_DDP(DDP):
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

    def get_tot_anoms(self):
        return sum([
            sum(self.module.data.ys[i]) \
            for i in range(self.module.data.T)
        ])

    def get_data(self):
        self.module.data.serialize()
        return self.module.data

    def clear_data(self):
        del self.module.data

    def train(self, mode=True):
        self.module.train(mode=mode)

    '''
    Put different data on workers
    '''
    def load_new_data(self, loader, kwargs):
        print(rpc.get_worker_info().name + ": Reloading %d - %d" % (kwargs['start'], kwargs['end']))
        
        jobs = kwargs.pop('jobs')
        self.module.data = loader(jobs, **kwargs)
        return True

    '''
    Combines data from other workers into one graph
    '''
    def student_becomes_master(self, others, reducer):
        futs = [
            _remote_method_async(GAE_DDP.get_data, rref)
            for rref in others
        ]

        datas = [self.module.data] + [f.wait() for f in futs]
        self.module.data = reducer(datas)


    '''
    Given node embeddings, return edge likelihoods for 
    all subgraphs held by this model, as well as edge names

    If detailed, uses full string for edge
    Else, just returns edge label 
    '''
    def decode_test(self, zs, detailed, single=False):
        edges = []

        with torch.no_grad():
            for i in range(self.module.data.T):
                if single:
                    z = zs
                else:
                    z = zs[i]
                
                scores = self.decode(self.module.data.eis[i], z)
                names = self.module.data.ys[i] if not detailed \
                    else self.module.data.format_edgelist(i)

                for j in range(len(names)):
                    edges.append(
                        (scores[j].item(), names[j])
                    )

        return edges

    def decode(self, e,z):
        src,dst = e 
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    def nll(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss
    
    '''
    Rather than sending edge index to master, calculate loss 
    on workers all at once 
    '''
    def calc_loss(self, zs, nratio, pred):
        # Uses masked val edges if module is set to eval()
        if self.module.training:
            partition = self.module.data.tr 
        else:
            partition = self.module.data.va 

        # Generate negative edges
        p,n,z = g.link_prediction(
            self.module.data, partition, zs,
            include_tr = not self.module.training, 
            nratio=nratio 
        )

        if pred: 
            p,n,z = p[1:], n[1:], z[:-1]
        
        T = len(z)

        p_scores = []
        n_scores = []

        for i in range(T):
            p_scores.append(self.decode(p[i], z[i]))
            n_scores.append(self.decode(n[i], z[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        if self.module.training:
            loss = self.nll(p_scores, n_scores)
            return loss

        else: 
            return p_scores, n_scores


'''
Called by worker processes to initialize their models
The RRefs to these models are then passed to the TGCN
'''
def get_remote_gae(h_dim, loader, load_args): 
    model = R_GAE(
        h_dim, loader, load_args
    )

    return GAE_DDP(model)    


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

        self.len_from_each = []


    '''
    Only need to tell workers which mask to use on the data, nothing else
    is needed. 
    '''
    def forward(self, mask_enum, include_h=False, h_0=None, no_grad=False):
        futs = self.encode(mask_enum, no_grad)

        # Run through RNN as embeddings come in 
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        for f in futs:
            z, h_0 = self.gru(
                torch.tanh(f.wait()),
                h_0, include_h=True
            )

            zs.append(z)

        # May as well do this every time, not super expensive
        self.len_from_each = [
            embed.size(0) for embed in zs
        ]

        if include_h:
            return torch.cat(zs, dim=0), h_0 
        else:
            return torch.cat(zs, dim=0)

    '''
    Tell each remote GCN to encode their data. Data lives there to minimise 
    net traffic 
    '''
    def encode(self, mask_enum, no_grad):
        embed_futs = []
        
        for i in range(self.num_workers):    
            embed_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.gcns[i % self.num_workers],
                    mask_enum, no_grad
                )
            )

        return embed_futs

    '''
    Has the distributed models score and label all of their edges
    '''
    def score_edges(self, zs, detailed=False, single=False):
        start = 0 
        futs = []

        for i in range(self.num_workers):
            if single:
                futs.append(
                    _remote_method_async(
                        GAE_DDP.decode_test,
                        self.gcns[i],
                        zs,
                        detailed=detailed,
                        single=single
                    )
                )
            else:
                end = start + self.len_from_each[i]
                futs.append(
                    _remote_method_async(
                        GAE_DDP.decode_test,
                        self.gcns[i],
                        zs[start : end],
                        detailed=detailed
                    )
                )
                start = end 

        # Concatenate all returned edge lists/scores
        rets = []
        for f in futs:
            rets += f.wait() 

        tot_anoms = sum(
            [_remote_method(
                GAE_DDP.get_tot_anoms, rref
            ) for rref in self.gcns]
        )    

        return rets, tot_anoms


    '''
    Returns either NLL or just edge scores if the model is in eval mode
    '''
    def __loss_or_score(self, zs, nratio, pred):
        futs = []
        start = 0 
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            
            futs.append(
                _remote_method_async(
                    GAE_DDP.calc_loss,
                    self.gcns[i],
                    zs[start : end], nratio, pred 
                )
            ) 
            start = end 

        return futs 


    '''
    Runs NLL on each worker machine given the generated embeds
    '''
    def loss_fn(self, zs, nratio=1, pred=True):
        assert self.training, \
            "Module must be in training mode for loss_fn to work as expected"

        futs = self.__loss_or_score(zs, nratio, pred)
        return torch.stack([f.wait() for f in futs]).mean() 

    '''
    Gets edge scores from dist modules. Assumes model is in eval mode
    '''
    def score_fn(self, zs, nratio=1, pred=True):
        assert not self.training, \
            "Module must be in evaluation mode for score_fn to work as expected"

        futs = self.__loss_or_score(zs, nratio, pred)
        pos, neg = [], []
        for f in futs:
            p,n = f.wait()
            pos.append(p)
            neg.append(n)

        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)


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

    
    '''
    Propogate to all workers
    '''
    def train(self, mode=True):
        super(TGCN, self).train() 
        [_remote_method(
            GAE_DDP.train,
            self.gcns[i],
            mode=mode
        ) for i in range(self.num_workers)]

    def eval(self):
        super(TGCN, self).train(False)
        [_remote_method(
            GAE_DDP.train,
            self.gcns[i],
            mode=False
        ) for i in range(self.num_workers)]