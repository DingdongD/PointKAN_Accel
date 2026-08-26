import numpy as np
import pdb

typename = 'fusionarch'
filename = 'simulation_'+typename+'.log'
sa_layerid = 0

slicenum = 8
corenum = 16
arr = None

class struct:
    def __init__(self):
        self.centroid = 0
        self.allocTime = 0
        self.releaseTime = 0

with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '[FC - input centroids]' in line:
            sa_layerid += 1
            if sa_layerid > 1:
                for sid in range(slicenum):
                    for cid in range(corenum):
                        print('{},{},{},{},{},{}'.format(sa_layerid-1, sid, cid, arr[sid][cid].centroid, arr[sid][cid].allocTime, arr[sid][cid].releaseTime))
            arr = [[struct() for i in range(corenum)] for j in range(slicenum)]
        elif '[Core Allocation]' in line:
            centroid = int(line[36:line.find('and')-1])
            sid = int(line[line.rfind('\'')-1])
            cid = int(line[line.rfind('re')+3:line.rfind('at')-1])
            allocTime = int(line[line.rfind('le')+3:-2])
            arr[sid][cid].centroid = centroid
            arr[sid][cid].allocTime = allocTime
        elif 'is released' in line:
            sid_startpos = line.find('ce')+3
            sid = int(line[sid_startpos:sid_startpos+line[sid_startpos:].find(' ')])
            cid_startpos = sid_startpos+line[sid_startpos:].find('re')+3
            cid = int(line[cid_startpos:line.find('is')-1])
            releaseTime = int(line[line.rfind('le')+3:])
            arr[sid][cid].releaseTime = releaseTime
    
    sa_layerid += 1
    for sid in range(slicenum):
        for cid in range(corenum):
            print('{},{},{},{},{},{}'.format(sa_layerid-1, sid, cid, arr[sid][cid].centroid, arr[sid][cid].allocTime, arr[sid][cid].releaseTime))
    


def show_arr():
    for sid in range(slicenum):
        print('sliceid:', sid)
        for cid in range(corenum):
            print('coreid: {}, alloc: {}, rel: {}'.format(cid, arr[sid][cid].centroid, arr[sid][cid].allocTime, arr[sid][cid].releaseTime))