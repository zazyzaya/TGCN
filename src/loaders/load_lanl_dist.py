import torch 
from torch_geometric.data import Data 

DATE_OF_EVIL_LANL = 150885
FILE_DELTA = 10000
LANL_FOLDER = '/mnt/raid0_24TB/isaiah/code/TGCN/src/data/split_LANL/'
RED_LOG = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/redteam.txt'

class LANL_Data(Data):
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
            src = 'C' + str(ei[0, i]])
            dst = 'C' + str(ei[1, i]])
            ts = self.slices[t]

            if 'ys' in self:
                anom = '' if self.ys[t][i] == 1 \
                        else 'ANOMALOUS'
            else:
                anom = ''

            edges.append('%s --> %s\t@ %s %s' % (src, dst, ts, anom))

        return edges

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files 
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156658, delta=1000, is_test=False):
    start_f = str(start - (start % FILE_DELTA)) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    # Helper functions (trims the 'C' off of comp ids)
    fmt_line = lambda x : (int(x[0]), int(x[1][1:]), int(x[2][1:]))
    def get_next_anom(rf):
        line = rf.readline().split(',')
        return fmt_line([line[0], line[3], line[4]])

    # If we're testing for anomalous edges, get the first anom that
    # will appear in this range (usually just the first one, but allows
    # for checking late time steps as well)
    if is_test:
        rf = open(RED_LOG, 'r')
        rf.readline() # Skip header
        
        next_anom = get_next_anom(rf)
        while next_anom[0] < start:
            next_anom = get_next_anom(rf)

    # Just uses the actual computer number as nids as this is to 
    # be used across distro'd processes to load small chunks at a time
    # and must match what other procs call the nodes
    edges = {}