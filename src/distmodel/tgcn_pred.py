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

        # The data for the next ei lives in each R_GAE for calculating loss
        # but we don't actually need the embeddings for it. 
        for i in range(self.data.T-1):
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

I'm just assuming the model will always be predictive, so 
I'm cleaning this up a bit
'''
class GAE_DDP(DDP):
    def get_tot_anoms(self):
        return sum([
            sum(self.module.data.ys[i]) \
            for i in range(self.module.data.T-1)
        ])

    def get_ys(self):
        return self.module.ys[:-1]

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
    all subgraphs held by this model

    If detailed, uses full string for edge
    Else, just returns edge label 
    '''
    def decode_all(self, zs):
        labels = []
        scores = []

        with torch.no_grad():
            for i in range(self.module.data.T-1):    
                scores.append(
                    self.decode(
                        self.module.data.eis[i+1], 
                        zs[i]
                    )
                )

                labels += self.module.data.ys[i+1]

        return torch.cat(scores, dim=0), torch.tensor(labels)

    '''
    Generates human-readable labels for all edges held by this worker
    that are decodable (edges[1:])
    '''
    def get_names(self):
        names = []
        for i in range(1,self.module.data.T):
            names += self.module.data.format_edgelist(i)

        return names

    def decode(self, e,z):
        src,dst = e 
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    def nll(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5
    
    def __get_neg_edges(self, z, partition, nratio):
        p,n,_ = g.link_prediction(
            self.module.data, self.module.data.p_fns[partition],
            z, start=1, nratio=nratio
        )

        T = len(z)
        assert not T > len(p), \
            rpc.get_worker_info().name + ' recieved more embeddings than subgraphs'
        assert not T < len(p), \
            rpc.get_worker_info().name + ' recieved fewer embeddings than subgraphs'

        return p,n

    '''
    Same as running calc loss in eval mode, but scores all nodes
    Assumes zs are already adjusted so z[0] predicts edge[1]
    '''
    def score_edges(self, z, partition, nratio):
        p,n = self.__get_neg_edges(z, partition, nratio)
        p_scores = []
        n_scores = []

        for i in range(len(z)):
            p_scores.append(self.decode(p[i], z[i]))
            n_scores.append(self.decode(n[i], z[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    '''
    Rather than sending edge index to master, calculate loss 
    on workers all at once 
    '''
    def calc_loss(self, z, partition, nratio):
        # First get edge scores
        p_scores, n_scores = self.score_edges(z, partition, nratio)

        # Then run NL loss on them
        return self.nll(p_scores, n_scores)


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

        '''
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
        '''
        zs = [f.wait() for f in futs]

        # May as well do this every time, not super expensive
        self.len_from_each = [
            embed.size(0) for embed in zs
        ]

        zs, h_0 = self.gru(
            torch.tanh(torch.cat(zs, dim=0)),
            h_0, include_h=True
        )

        if include_h:
            return zs, h_0 
        else:
            return zs

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
    def score_all(self, zs, detailed=False):
        start = 0 
        readable=None
        futs = []

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    GAE_DDP.decode_all,
                    self.gcns[i],
                    zs[start : end]
                )
            )
            start = end 

        # Concatenate all returned edge lists/scores
        scores, labels = [],[]
        for f in futs:
            score, label = f.wait() 
            scores.append(score)
            labels.append(label)
             
        scores = torch.cat(scores, dim=0).squeeze()
        labels = torch.cat(labels, dim=0).squeeze()

        if detailed:
            futs = [
                _remote_method_async(
                    GAE_DDP.get_names,
                    self.gcns[i]
                )
                for i in range(self.num_workers)
            ]

            readable = sum(
                [f.wait() for f in futs],
                []
            )

        return scores, labels, readable


    '''
    Runs NLL on each worker machine given the generated embeds
    '''
    def loss_fn(self, zs, partition, nratio=1):
        futs = []
        start = 0 

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            
            futs.append(
                _remote_method_async(
                    GAE_DDP.calc_loss,
                    self.gcns[i],
                    zs[start : end], 
                    partition, nratio
                )
            ) 
            start = end 

        return torch.stack([f.wait() for f in futs]).mean() 

    '''
    Gets edge scores from dist modules, and negative edges
    '''
    def score_edges(self, zs, partition, nratio=1):
        futs = []
        start = 0 

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            
            futs.append(
                _remote_method_async(
                    GAE_DDP.score_,
                    self.gcns[i],
                    zs[start : end], 
                    partition, nratio
                )
            ) 
            start = end 

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