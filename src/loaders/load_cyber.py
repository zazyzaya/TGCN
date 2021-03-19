import json
import os 
from tqdm import tqdm

import torch 
from torch_geometric.data import Data

from .load_utils import edge_tv_split

class CyData(Data):
    def __init__(self, **kwargs):
        super(CyData, self).__init__(**kwargs)

        # Getter methods so I don't have to write this every time
        self.tr = lambda t : self.eis[t][:, self.masks[t]] 
        self.va = lambda t : self.eis[t][:, ~self.masks[t]]

        # Assumes test sets are unmasked (online classification)
        self.te = lambda t : self.eis[t] 
        self.all = self.te # For readability


    '''
    Makes it a little easier to format an edge list for printing
    '''
    def format_edgelist(self, t):
        ei = self.eis[t]
        edges = []

        for i in range(ei.size(1)):
            src = self.node_map[ei[0, i]]
            dst = self.node_map[ei[1, i]]
            
            if 'ts' in self:
                ts = self.ts[t][i]
            elif 'slices' in self:
                ts = self.slices[t]
            else:
                ts = ''

            if 'ys' in self:
                anom = '' if self.ys[t][i] == 1 \
                        else 'ANOMALOUS'
            else:
                anom = ''
            
            arrow = '-->' if 'efs' not in self \
                    else '-( %s )->' % self.efs[t][i]

            edges.append('%s %s %s\t@ %s %s' % (src, arrow, dst, ts, anom))

        return edges


# Last timestamp before malware in system
DATE_OF_EVIL_PICO = "2019-07-19T18:07:59.460425Z"
DATE_OF_EVIL_LANL = 150885

def pico_file_loader(fname, keep=['client', 'ts', 'service', 'success', 'id.orig_h']):
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
def pico_logs_to_graph(logs, delta, whitelist=[]):
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
    ef = []

    # Kind of cheating because DNS changes these, but if possible
    # when an admin account does something, it's helpful to know
    # what computer it was from for comparison to the Red Log 
    ip_map = {
        '27': 'RND-WIN10-2',
        '29': 'RND-WIN10-1',
        '30': 'HR-WIN7-2',
        '152': 'HR-WIN7-1',
        '160': 'SUPERSECRETXP',
        '5': 'CORP-DC',
        '100': 'PFSENSE'
    }

    for i in tqdm(range(len(logs))):
        l = logs[i]

        # First parse out client 
        client = l['client'].split('/')[0] # Don't really care about instance 
        client = client.upper()

        # Update ip map if needed (because going sequentially DNS updates
        # can happen as we see them)
        ip = l['id.orig_h'].split('.')[-1]
        if '$' in client and ip_map[ip] != client.replace('$', ''):
            ip_map[ip] = client.replace('$', '')


        if tr_set and l['ts'] == DATE_OF_EVIL_PICO:
            tr_set = False 
            tr_set_partition_end = i // delta
        
        skip = False
        for wl in whitelist:
            if wl in l['service'].upper():
                skip = True
        if skip:
            continue
    
        if 'ADMIN' in client:
            client = client.replace('ADMINISTRATOR', 'ADMIN') # Shorten a bit
            
            try:
                maybe_comp = ip_map[ip]
            except:
                maybe_comp = 'UNK'

            client += '@' + maybe_comp

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
        ef.append('succ' if l['success'] else 'fail')

        # Create new partition for next set of edges
        if (i+1) % delta == 0:
            eis.append([src,dst])
            src = []
            dst = []

            edge_times.append(ts)
            ts = []

            efs.append(ef)
            ef = []

    return make_data_obj(eis, cl_to_id, tr_set_partition_end, ts=edge_times, efs=efs)


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

def load_pico(delta=6):
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

    return pico_logs_to_graph(logs, delta)


'''
Default start and end times are sort of arbitrary
end time is right around the 5th anomalous event

TODO
'''
def load_lanl(start=140000, end=156658, delta=1000):
    F_LOC = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/auth.txt'
    F_LOC_R = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/redteam.txt'

    '''
    0 time,                       <-- pay attention to deliniate train/test
    1 source user@domain,         
    2 destination user@domain,    
    3 source computer,            <-- src node
    4 destination computer,       <-- dst node
    5 authentication type,        <-- edge feat
    6 logon type,                 <-- edge feat
    7 authentication orientation, <-- edge feat
    8 success/failure
    '''

    # Helper functions for formatting/readability
    get_ts = lambda x : int(x[0])
    get_src = lambda x : x[3]
    get_dst = lambda x : x[4]
    
    def get_or_add(s,d,c):
        if s not in d:
            d[s] = c[0]
            c[0] += 1
        return d[s]
    
    def add_edge(et, edict):
        l = et[-1]
        et = (et[0], et[1])

        if et in edict:
            edict[et] = min(l, edict[et])
        else:
            edict[et] = l

    def get_next_anom(rf):
        line = rf.readline().split(',')
        ts = get_ts(line)
        src = line[2]
        dst = line[3][:-1] # Strip out newline

        return ts, src, dst

    f = open(F_LOC, 'r')
    red = open(F_LOC_R, 'r')

    train = True
    capturing = False
    te_starts = 0

    #Skip header
    line = f.readline()
    line = f.readline()

    # Skip header
    red.readline()
    next_anom = get_next_anom(red)

    progress = tqdm()

    eis = []
    edict = {}

    # Counters in arrays so we can pass by reference
    node_dict = {}
    ndc = [0] 
    
    # Edge features 
    times = []
    weights = []

    ys = []
    y = []

    ticks = 0
    cur_time = 0

    while(line):    
        progress.update(1)
        line = line.split(',')
        
        ts = get_ts(line)
        # Good god there's a lot of logs; cut it off after
        # a few anomalies are captured
        if ts < start:
            line = f.readline()
            continue
        else:
            if not capturing:
                cur_time = ts
                capturing = True

        if ts >= end:
            break

        # Create new graph if enough time has passed
        if ts != cur_time:
            ticks += ts - cur_time 

            if ticks % delta == 0 or ts-cur_time >= delta:
                ei = list(zip(*edict.keys()))
                eis.append(ei)

                y = list(edict.values())
                ys.append(y)

                edict = {}
                times.append(ts-delta)
            
            cur_time = ts
        
        # Mark future time steps as test set if we hit the 
        # specified timestep where anomalies start
        if ts == DATE_OF_EVIL_LANL and train:
            train = False 
            te_starts = len(eis)
            print(line) 
            print(next_anom)
            print(ts)

        # Can now deal with the data on this line 
        src = get_src(line)
        dst = get_dst(line)

        # Skip logs with missing info 
        if '?' in src+dst:
            line = f.readline()
            continue

        label = 1
        if not train and ts == next_anom[0]:
            if src == next_anom[1] and dst == next_anom[2]:
                label = 0
                print('label is 0')
                next_anom = get_next_anom(red)
 
        e_tuple = [
            get_or_add(src, node_dict, ndc),
            get_or_add(dst, node_dict, ndc),
            label
        ]

        add_edge(e_tuple, edict)
        line = f.readline()

    f.close()   
    red.close()
    data = make_data_obj(
        eis, node_dict, te_starts, 
        slices=times, ys=ys
    )
    return data 

