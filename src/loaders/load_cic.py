import json 

import pandas as pd
import torch
from torch_geometric.data import Data 

from .load_utils import edge_tv_split

class CICData(Data):
    def __init__(self, **kwargs):
        super(CICData, self).__init__(**kwargs)

        # Tr/val only applies to t < te_starts
        self.tr = lambda t : self.eis[t][: self.masks[t][0]]
        self.va = lambda t : self.eis[t][: self.masks[t][1]]
        
        # Test set is just unseen time slices
        self.te = lambda t : self.eis[t] 


HOME = '/mnt/raid0_24TB/datasets/cic/iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/TrafficLabelling/'
SUFFIX = '.pcap_ISCX.csv'

# For easiser loading
FMAP = {
    'M': 'Monday-WorkingHours',
    'T': 'Tuesday-WorkingHours',
    'W': 'Wednesday-WorkingHours',
    'H0': 'Thursday-WorkingHours-Morning-WebAttacks',
    'H1': 'Thursday-WorkingHours-Afternoon-Infiltration',
    'F0': 'Friday-WorkingHours-Morning',
    'F1': 'Friday-WorkingHours-Afternoon-PortScan',
    'F2': 'Friday-WorkingHours-Afternoon-DDos'
}

INV_FMAP = {v:k for (k,v) in FMAP.items()}

# Just hard coding this since it's easier
# Info from here: https://www.unb.ca/cic/datasets/ids-2017.html
TE_STARTS = {
    'M': 'nan',
    'T': 'nan',
    'W': 'nan',
    'H0': '9:20',
    'H1': '14:19',
    'F0': '10:02',
    'F1': '13:55',
    'F2': '15:56'
}

'''
Fname:  a code from FMAP
delta:  size of time slices (in minutes) (TODO unused rn)
end:    time stamp to end training set at. If None, first ts where
        anomalies are present
'''
def load_cic(fname='H0', delta=1, end='10:45', te_starts=None):
    assert fname in FMAP.keys() or fname in FMAP.values(), \
        'fname must be either a key or value from\n%s' % json.dumps(FMAP, indent=2)
        
    if fname in FMAP:
        fname = FMAP[fname]
    
    te_starts = te_starts if te_starts else TE_STARTS[INV_FMAP[fname]]
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

    times = []
    tss = []

    node_map = {}
    nid = 0

    masks = []

    te_start_marked = False

    # Helper functions
    isnan = lambda x : x != x
    def get_or_add(k, id):
        if k in node_map:
            return node_map[k]
        
        node_map[k] = nid
        nid += 1
        return nid-1

    curtime = df[0]['ts']
    src, dst = None 
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
        ts = df['ts'][i]
        if curtime != ts:
            curtime = ts 

            eis.append(torch.tensor(ei))
            ei = [[],[]]
            masks.append(edge_tv_split(eis[-1]))

            times.append(tss)
            tss = []

            labels.append(torch.tensor(label_t))
            label_t = []

        # Every point up until now is clean data, 
        # after may be anomalous
        if ts == te_starts and not te_start_marked:
            te_starts_idx = len(eis)

        src = get_or_add(src)
        dst = get_or_add(dst)
        y = 1 if y == 'BENIGN' else 0

        ei[0].append(src)
        ei[1].append(dst)
        tss.append(ts)
        label_t.append(y)

        i+=1
    
    num_nodes = len(node_map)
    return CICData(
        x=torch.eye(num_nodes),
        y=labels,
        eis=eis,
        te_starts=te_starts_idx,
        node_map=node_map,
        masks=masks
    )