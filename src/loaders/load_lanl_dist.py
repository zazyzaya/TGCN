import os 
import pickle 
import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

from .load_utils import edge_tv_split

DATE_OF_EVIL_LANL = 150885
FILE_DELTA = 10000
LANL_FOLDER = '/mnt/raid0_24TB/isaiah/code/TGCN/src/data/split_LANL/'
RED_LOG = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/redteam.txt'

class LANL_Data(Data):
    def __init__(self, **kwargs):
        super(LANL_Data, self).__init__(**kwargs)

        # Getter methods so I don't have to write this every time
        self.tr = lambda t : self.eis[t][:, self.masks[t]] 
        self.va = lambda t : self.eis[t][:, ~self.masks[t]]

        self.tr_w = lambda t : self.ews[t][self.masks[t]]
        self.va_w = lambda t : self.ews[t][~self.masks[t]]

        # Assumes test sets are unmasked (online classification)
        self.te = lambda t : self.eis[t] 
        self.te_w = lambda t : self.ews[t]

        # For readability
        self.all = self.te 
        self.all_w = self.te_w


    '''
    Makes it a little easier to format an edge list for printing
    '''
    def format_edgelist(self, t):
        ei = self.eis[t]
        edges = []

        for i in range(ei.size(1)):
            src = self.node_map[ei[0, i]]
            dst = self.node_map[ei[1, i]]
            ts = self.slices[t]

            if self.ys:
                anom = '' if self.ys[t][i] == 0 \
                        else 'ANOMALOUS'
            else:
                anom = ''

            edges.append('%s --> %s\t@ %s %s' % (src, dst, ts, anom))

        return edges


def make_data_obj(eis, tr_set_partition_end, **kwargs):
    # Known value for LANL
    cl_cnt = 17684

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
    data = LANL_Data(
        x=x, 
        eis=eis_t,
        masks=splits,
        te_starts=tr_set_partition_end,
        num_nodes=cl_cnt,
        T=len(eis),
        #node_map=node_map,
        **kwargs
    )
    return data

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files 
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=1000, is_test=False):
    cur_slice = start - (start % FILE_DELTA)
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    times = []

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    # Helper functions (trims the trailing \n)
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2][:-1]))
    def get_next_anom(rf):
        line = rf.readline().split(',')
        return (int(line[0]), line[2], line[3])

    # For now, just keeps one copy of each edge. Could be
    # modified in the future to add edge weight or something
    # but for now, edges map to their anomaly value (1 == anom, else 0)
    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1)
        else:
            edges_t[et] = (is_anom, 1)

    def is_anomalous(src, dst, anom):
        src = node_map[src]
        dst = node_map[dst]
        return src==anom[1] and dst==anom[2][:-1]

    # If we're testing for anomalous edges, get the first anom that
    # will appear in this range (usually just the first one, but allows
    # for checking late time steps as well)
    if is_test:
        rf = open(RED_LOG, 'r')
        rf.readline() # Skip header
        
        next_anom = get_next_anom(rf)
        while next_anom[0] < start:
            next_anom = get_next_anom(rf)
    else:
        next_anom = (-1, 0,0)


    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_starts = 0
    tot_anoms = 0
    anom_marked = False
    keep_reading = True

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime 
    while keep_reading:
        while line:
            l = line.split(',')
            
            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts 
                continue
            
            ts, src, dst = fmt_line(l)
            et = (src,dst)

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts 

            # Split edge list if delta is hit 
            # (assumes no missing timesteps in the log files)
            if (curtime != ts and (curtime-ts) % delta == 0) or ts >=end:
                ei = list(zip(*edges_t.keys()))
                edges.append(ei)

                y,ew = list(zip(*edges_t.values()))
                ews.append(
                    torch.sigmoid(torch.tensor(ew, dtype=torch.float))
                )
                if is_test:
                    ys.append(y)

                edges_t = {}
                times.append(str(curtime) + '-' + str(ts-1))
                curtime = ts 

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False 
                    break 

            # Mark edge as anomalous if it is 
            if ts == next_anom[0] and is_anomalous(src, dst, next_anom):
                add_edge(et, is_anom=1)
                next_anom = get_next_anom(rf)
                tot_anoms += 1

                # Mark the first timestep with anomalies as test set start
                if not anom_marked:
                    anom_marked = True
                    anom_starts = len(edges)

            else:
                add_edge(et)

            line = in_f.readline()

        in_f.close() 
        cur_slice += FILE_DELTA 

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            break
    
    in_f.close() 
    
    if is_test:
        rf.close() 

    if not is_test:
        anom_starts = len(edges)

    ys = ys if is_test else None
    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, 
        anom_starts,  
        slices=times,
        ys=ys,
        tot_anoms=tot_anoms,
        ews=ews
    )