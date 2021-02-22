import json
import os 
from tqdm import tqdm

import torch 
from torch_geometric.data import Data

from utils import edge_tv_split

class CyData(Data):
    def __init__(self, **kwargs):
        super(CyData, self).__init__(**kwargs)

        # Getter methods so I don't have to write this every time
        self.tr = lambda t : self.eis[t][:, self.masks[t]] 
        self.va = lambda t : self.eis[t][:, ~self.masks[t]]
        
        # Return time before or after train/test split
        self.tr_slice = lambda : self.eis[:self.te_starts]
        self.te_slice = lambda : self.eis[self.te_starts:]

        # Used for negative sampling
        self.tr_range = range(0, self.te_starts)
        self.te_range = range(self.te_starts, self.T)

        # Assumes test sets are unmasked (online classification)
        self.te = lambda t : self.eis[t] 


    '''
    Makes it a little easier to format an edge list for printing
    '''
    def format_edgelist(self, t):
        ei = self.eis[t]
        edges = []

        for i in range(ei.size(1)):
            src = self.node_map[ei[0, i]]
            dst = self.node_map[ei[1, i]]
            ts = self.ts[t][i]
            
            edges.append('%s --> %s  @\t%s' % (src, dst, ts))

        return edges


# Last timestamp before malware in system
DATE_OF_EVIL_PICO = "2019-07-19T18:07:59.460425Z"
DATE_OF_EVIL_LANL = 150885

def pico_file_loader(fname, keep=['client', 'ts', 'service']):
    # Not really in the right schema to just use json.loads on the 
    # whole thing. Each line is its own json object
    with open(fname, 'r') as f:
        lines = f.read().split('\n')
        logs = [json.loads(l) for l in lines if len(l) > 1]

    # Filter out noisey logs. Only care about TGS kerb logs (for now)
    unflogs = [l for l in logs if 'request_type' in l.keys()]
    
    # Get rid of extranious data, and make sure required data exists
    logs = []
    for l in unflogs:
        try:
            logs.append({k:l[k] for k in keep})
        except KeyError as e:
            continue 

    return logs 


'''
Given several JSON objects representing single hour of auth data,
generate temporal graphs with delta-hour long partitions
'''
def pico_logs_to_graph(logs, delta=6, whitelist=[]):
    '''
    (Useful) Kerb logs have the following structure:
                                   (where they access from)
    client:   USR (or) COMPUTER$ / INSTANCE . subdomain . subdomain (etc)

             (optional)    (the computer)                     (optional)
    service:  service   /  TOP LEVEL .    sub domain . (etc)  @  realm  

    Worth noting, service names for computers are in all caps, client names are.. varied.
    To be safe, capitalize everything
    '''
    tr_set = True
    tr_set_partition_end = 0

    cl_to_id = {}  # Map of string names to numeric 0-|N|
    cl_cnt = 0         # Cur id

    src = []        # User machine or name
    dst = []        # Service machine
    
    ts = []
    edge_times = [] 

    eis = []
    efs = []

    timer=0

    for i in tqdm(range(len(logs))):
        l = logs[i]

        if tr_set and l['ts'] == DATE_OF_EVIL_PICO:
            tr_set = False 
            tr_set_partition_end = i // delta
        
        skip = False
        for wl in whitelist:
            if wl in l['service'].upper():
                skip = True
        if skip:
            continue

        # First parse out client 
        client = l['client'].split('/')[0] # Don't really care about instance 
        client = client.upper()
        
        if client in cl_to_id:
            src.append(cl_to_id[client])
        else:
            src.append(cl_cnt)
            cl_to_id[client] = cl_cnt
            cl_cnt += 1
        
        # Then parse out server & service 
        srv = l['service']#.split('/')
        
        '''
        # As far as I can tell, just 2 cases:
        if len(srv) == 1:
            server = srv[0].upper()
            server = server.split('@')[0] # ignore realm if it exists

        else:
            server = srv[1].split('.')[0].upper() # Only care about top-level (also slices out realm)
        '''
        server = srv.split('.')[0].upper()

        # Add in id of server 
        if server in cl_to_id:
            dst.append(cl_to_id[server])
        else:
            dst.append(cl_cnt)
            cl_to_id[server] = cl_cnt
            cl_cnt += 1

        ts.append(l['ts'])

        # Create new partition for next set of edges
        if (i+1) % delta == 0:
            eis.append([src,dst])
            src = []
            dst = []

            edge_times.append(ts)
            ts = []

    return make_data_obj(eis, cl_to_id, tr_set_partition_end, ts=edge_times)


def make_data_obj(eis, cl_to_id, tr_set_partition_end, **kwargs):
    cl_cnt = max(cl_to_id.values())
    
    node_map = [None] * (max(cl_to_id.values()) + 1)
    for k,v in cl_to_id.items():
        node_map[v] = k

    # No node feats really
    x = torch.eye(cl_cnt+1)
    
    # Build time-partitioned edge lists
    eis_t = []
    splits = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        
        # Add val mask for all time slices in training set
        if i < tr_set_partition_end:
            splits.append(edge_tv_split(ei)[0])

    # Finally, return Data object
    data = CyData(
        x=x, 
        eis=eis_t,
        masks=splits,
        te_starts=tr_set_partition_end,
        num_nodes=x.size(0),
        T=len(eis),
        node_map=node_map,
        **kwargs
    )
    return data

def load_pico():
    F_LOC = '/mnt/raid0_24TB/datasets/pico/bro/'
    days = [os.path.join(F_LOC,d) for d in os.listdir(F_LOC)]
    days.sort()

    logs = []
    for d in days:
        kerb_logs = [os.path.join(d, l) for l in os.listdir(d) if 'kerb' in l]
        kerb_logs.sort()

        list_o_logs = [pico_file_loader(l) for l in kerb_logs]
        for l in list_o_logs:
            logs += l

    return pico_logs_to_graph(logs)


'''
Default start and end times are sort of arbitrary
end time is right around the 5th anomalous event

TODO
'''
def load_lanl(start=140000, end=156658, delta=1000):
    F_LOC = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/auth.txt'

    # Counters in arrays so we can pass by reference
    node_dict = {}
    ndc = [0] 

    feat_dict = {}
    fdc = [0] 

    src = []
    dst = []
    feats = []


    '''
    0 time,                       <-- pay attention to deliniate train/test
    1 source user@domain,         <-- node (ignore domain)
    2 destination user@domain,    <-- ditto
    3 source computer,            
    4 destination computer,
    5 authentication type,        <-- edge feat
    6 logon type,                 <-- edge feat
    7 authentication orientation, <-- edge feat
    8 success/failure
    '''
    #slice_usr = lambda x : x.split('@')[0].upper()
    slice_usr = lambda x : x.upper()

    def get_or_add(s,d,c):
        if s not in d:
            d[s] = c[0]
            c[0] += 1
        return d[s]
            

    f = open(F_LOC, 'r')
    train = True
    tr_set_end = 0

    #Skip header
    line = f.readline()
    line = f.readline()
    progress = tqdm()

    e_set = {}
    ect = 0
    weights = []

    while(line):    
        progress.update(1)
        line = line.split(',')

        # Good god there's a lot of logs; cut it off after
        # a few anomalies are captured
        if int(line[0]) < start:
            line = f.readline()
            continue

        if int(line[0]) >= end:
            break

        src_u = slice_usr(line[1])
        dst_u = slice_usr(line[2])
        feat = ', '.join([line[5], line[8]]).upper()[:-1] # Remove trailing \n

        # Skip logs with missing info 
        if '?' in src_u+dst_u+feat:
            line = f.readline()
            continue 

        # Prevent duplicate edges
        e_tuple = (
            get_or_add(src_u, node_dict, ndc),
            get_or_add(dst_u, node_dict, ndc),
            get_or_add(feat, feat_dict, fdc)
        )

        if e_tuple not in e_set:
            if train:
                if int(line[0]) >= DATE_OF_EVIL_LANL:
                    train = False
                else:
                    tr_set_end += 1

            e_set[e_tuple] = ect
            ect += 1
            weights.append(1)

            src.append(e_tuple[0])
            dst.append(e_tuple[1])
            feats.append(e_tuple[2])
        else:
            weights[e_set[e_tuple]] += 1

        line = f.readline()

    f.close()   
    data = make_data_obj(src, dst, feats, node_dict, feat_dict, tr_set_end)
    data.edge_weight = torch.sigmoid(torch.tensor(weights).float())
    return data 