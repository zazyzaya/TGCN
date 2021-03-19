import os 
from tqdm import tqdm

# LANL is so huge it's prohibitively expensive to scan it for 
# edges of later time steps. To remedy this (and make it easier
# for the distro models to load data later on) I have split it into
# files containing 10,000 seconds each 

SRC = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/auth.txt'
DST = '/mnt/raid0_24TB/isaiah/code/TGCN/src/data/split_LANL/'

last_time = 1
cur_time = 0
DELTA = 10000

f_in = open(SRC,'r')
f_out = open(DST + str(cur_time) + '.txt', 'w+')

line = f_in.readline() # Skip headers
line = f_in.readline()

prog = tqdm(desc='Seconds parsed', total=5011199)

# Really only care about time stamp, and src/dst computers
# Hopefully this saves a bit of space when replicating the huge
# auth.txt flow file
fmt_line = lambda x : ('%s,%s,%s\n' % (x[0],x[3],x[4]), int(x[0]))
while line:
    l, ts = fmt_line(line.split(','))
    
    if ts != last_time:
        prog.update(ts-last_time)
        last_time = ts

    # After ts progresses at least 10,000 seconds, make a new file
    if ts >= cur_time+DELTA:
        cur_time += DELTA
        f_out.close()
        f_out = open(DST + str(cur_time) + '.txt', 'w+')
    
    f_out.write(l)
    line = f_in.readline()

f_out.close()
f_in.close()