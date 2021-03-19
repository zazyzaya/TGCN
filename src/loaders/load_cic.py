import json 

import pandas as pd
import torch
from torch_geometric.data import Data 
from tqdm import tqdm 

from .load_utils import edge_tv_split

class CICData(Data):
    def __init__(self, **kwargs):
        super(CICData, self).__init__(**kwargs)

        # Tr/val only applies to t < te_starts
        self.tr = lambda t : self.eis[t][:, self.masks[t][0]]
        self.va = lambda t : self.eis[t][:, self.masks[t][1]]
        
        # Test set is just unseen time slices
        self.te = lambda t : self.eis[t] 
        self.all = self.te # For readability

    '''
    Combine two CICData members. Assumes self is entirely
    training data, and precedes o temporally
    '''
    def __add__(self, o):
        eis = self.eis + o.eis
        te_starts = self.T + o.te_starts
        T = self.T + o.T
        y = self.y + o.y 
        te_starts = self.T + o.te_starts
        node_map = o.node_map
        masks = self.masks + o.masks
        x = o.x

        return CICData(
            x=x,
            y=y,
            eis=eis,
            te_starts=te_starts,
            node_map=node_map,
            masks=masks,
            T=T
        )


HOME = '/mnt/raid0_24TB/datasets/cic/iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/TrafficLabelling/'
SUFFIX = '.pcap_ISCX.csv'

# For easiser loading
FMAP = {
    'M': 'Monday-WorkingHours',
    'T': 'Tuesday-WorkingHours',
    'W': 'Wednesday-workingHours', # lowercase w here for some reason?
    'H0': 'Thursday-WorkingHours-Morning-WebAttacks',
    'H1': 'Thursday-WorkingHours-Afternoon-Infilteration',
    'F0': 'Friday-WorkingHours-Morning',
    'F1': 'Friday-WorkingHours-Afternoon-PortScan',
    'F2': 'Friday-WorkingHours-Afternoon-DDos'
}

INV_FMAP = {v:k for (k,v) in FMAP.items()}

# Just hard coding this since it's easier
# Info from here: https://www.unb.ca/cic/datasets/ids-2017.html
TE_STARTS = {
    'M': 'nan',
    'T': '9:20',
    'W': '9:47',
    'H0': '9:20',
    'H1': '2:19',
    'F0': '10:02',
    'F1': '1:55',
    'F2': '3:56'
}

# They're so god damn inconsistent with timecodes. It's mindblowing
has_seconds = ['M']

'''
Fname:  a code from FMAP
delta:  size of time slices (in minutes) (TODO unused rn)
end:    time stamp to end training set at. If None, first ts where
        anomalies are present
'''
def load_cic(fname='H0', delta=1, end='10:00', te_starts=None, node_map={}):
    
    assert fname in FMAP.keys() or fname in FMAP.values(), \
        'fname must be either a key or value from\n%s' % json.dumps(FMAP, indent=2)
        
    if fname in FMAP:
        fname = FMAP[fname]
    
    te_starts = te_starts if te_starts else TE_STARTS[INV_FMAP[fname]]
    clip_seconds = INV_FMAP[fname] in has_seconds
    fname = HOME + fname + SUFFIX

    df = pd.read_csv(
        fname, header=0, encoding='latin',
        usecols=[' Source IP', ' Destination IP', ' Timestamp', ' Label'],   
    )
    df.columns = ['src', 'dst', 'ts', 'y']

    labels = []
    label_t = []

    eis = []
    ei = [[],[]]

    nid = [len(node_map)] # To pass by reference

    ticks = 0
    masks = []

    te_start_marked = False
    te_starts_idx = float('inf')
    prog = tqdm(desc='Records parsed')

    # Helper functions
    isnan = lambda x : x != x
    fmt_time = lambda x : x[9:] if not clip_seconds else x[11:-3]

    def get_or_add(k):
        if k in node_map:
            return node_map[k]
        
        node_map[k] = nid[0]
        nid[0] += 1
        return nid[0]-1

    curtime = fmt_time(df['ts'][0])
    src, dst = None, None
    times = [curtime]

    # Note: for some reason the frame is larger
    # than the number of nodes, ending at idx 170365 (for H0)
    i=0
    while curtime != end:
        src = df['src'][i]
        dst = df['dst'][i]
        y = df['y'][i]

        # If both are nan the stream is over
        if isnan(src) and isnan(dst):
            break 
        # If one is nan.. not sure how to deal with that
        # just skip for now
        if isnan(src) or isnan(dst):
            i+=1
            continue

        # Hacky way of doing this right now. We assume time
        # deltas are just 1 min (smallest time inc) so any 
        # change is indicitive of a new slice
        ts = fmt_time(df['ts'][i])
        if curtime != ts:
            curtime = ts 
            ticks += 1
            prog.set_description(ts)

            if ticks % delta == 0:
                eis.append(torch.tensor(ei))
                ei = [[],[]]
                masks.append(edge_tv_split(eis[-1]))

                times.append(ts)

                labels.append(torch.tensor(label_t))
                label_t = []

        # Every point up until now is clean data, 
        # after may be anomalous
        if ts == te_starts and not te_start_marked:
            te_starts_idx = len(eis)
            te_start_marked = True

        src = get_or_add(src)
        dst = get_or_add(dst)
        y = 1 if y == 'BENIGN' else 0

        ei[0].append(src)
        ei[1].append(dst)
        label_t.append(y)

        i+=1
        prog.update(1)
    
    num_nodes = len(node_map)
    return CICData(
        x=torch.eye(num_nodes),
        y=labels,
        eis=eis,
        te_starts=te_starts_idx,
        node_map=node_map,
        masks=masks,
        T=len(eis)
    )