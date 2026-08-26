import pdb
import numpy as np
typename = 'fusionarch_shapenet_32x_scaledown'
filename = '../ours-fixdram/simulation_'+typename+'.log'
sa_layerid = -1

mode = 'fusionarch' # 'pointacc'

slicenum = 8 if mode == 'fusionarch' else 16
corenum = 16
layernum = 3
last_reltime_per_layer = [(0, 0) for i in range(layernum)] # (relTime, true_relTime)
mlp_macs = [[6144, 131072, 262144], [1073152, 1048576, 2097152], [8486912, 16777216, 67108864]] # pointnet++ cls
# mlp_macs = [[12288, 32768, 65536], [137216, 131072, 262144], [536576, 524288, 1048576], [2121728, 2097152, 4194304]]    # pointnet++ sem_seg
# mlp_macs = [[18432, 131072, 262144], [536576, 524288, 1048576], [8486912, 16777216, 67108864]]  # pointnet++ part_seg
# mlp_macs = [[167772160], [2726297600], [5452595200], [21642608640]]    # dgcnn cls
# mlp_macs = [[6207569920], [16441671680], [10905190400]]   # dgcnn part seg
# mlp_macs = [[7214202880], [16441671680], [10905190400]]   # dgcnn sem seg (s3dis)
mlp_macs = [[x/4 for x in a] for a in mlp_macs]
mlp_macs = [sum(x) for x in mlp_macs]
mac_array_size = 4*4
mlp_latency = [x/mac_array_size for x in mlp_macs]

arr = [[[[] for i in range(corenum)] for j in range(slicenum)] for k in range(layernum)]

delta = 0   # MLP cycles per layer

class struct:
    def __init__(self):
        self.coreid = 0
        self.centroid = 0
        self.allocTime = 0
        self.releaseTime = 0
        self.true_allocTime = 0
        self.true_relTime = 0
        # self.next = None
        # self.is_last = False
    def init(self, coreid, centroid, at):
        self.coreid = coreid
        self.centroid = centroid
        self.allocTime = at
        # self.next = None
        # self.is_last = False


with open(filename, 'r') as f:
    lines = f.readlines()
    print('layerid,sliceid,coreid,centroid,allocTime,true_allocTime,releaseTime,true_relTime')
    for line in lines:
        if '[FC - input centroids]' in line:
            sa_layerid += 1
            if sa_layerid > 0:
                flat_lst = [element for sublist1 in arr[sa_layerid-1] for sublist2 in sublist1 for element in sublist2]
                max_relTime = max(flat_lst, key=lambda x: x.releaseTime).releaseTime
                max_True_relTime = max(flat_lst, key=lambda x: x.true_relTime).true_relTime
                # for sid in range(slicenum):
                    # for cid in range(len(arr[sa_layerid-1][sid])):
                    #     if arr[sa_layerid-1][sid][cid].releaseTime > maxval:
                    #         maxval = arr[sa_layerid-1][sid][cid].releaseTime
                last_reltime_per_layer[sa_layerid-1] = (max_relTime, max_True_relTime)
        elif '[Core Allocation]' in line:
            centroid = int(line[36:line.find('and')-1])
            sid = int(line[line[:line.rfind('\'')].rfind(' ')+1:line.rfind('\'')])
            cid = int(line[line.rfind('re')+3:line.rfind('at')-1])
            allocTime = int(line[line.rfind('at')+3:-2]) if mode == 'pointacc' else int(line[line.rfind('le')+3:-2])
            # allocTime = int(line[line.rfind('le')+3:-2])
            obj = struct()
            obj.init(cid, centroid, allocTime)
            prev = len(arr[sa_layerid][sid][cid])-1
            if prev >= 0:
                true_allocTime = allocTime - arr[sa_layerid][sid][cid][prev].releaseTime + arr[sa_layerid][sid][cid][prev].true_relTime
            else:
                true_allocTime = allocTime - last_reltime_per_layer[sa_layerid-1][0] + last_reltime_per_layer[sa_layerid-1][1]
            obj.true_allocTime = true_allocTime
            arr[sa_layerid][sid][cid].append(obj)

            # if arr[sa_layerid][sid][cid].allocTime > 0:
            #     # core is reused
            #     # print('Attention - 1: lid: {}, sid: {}, cid: {}, centroid: {}, allocTime: {}'.format(sa_layerid, sid, cid, centroid, allocTime))
            #     prev = find_prev(sa_layerid, sid, cid)

            #     if prev > 0:
            #         obj = struct()
            #         obj.init(cid, centroid, allocTime)
            #         true_allocTime = allocTime - arr[sa_layerid][sid][prev].releaseTime + arr[sa_layerid][sid][prev].true_relTime
            #         arr[sa_layerid][sid].append(obj)
            #         if arr[sa_layerid][sid][cid].next is None:
            #             arr[sa_layerid][sid][cid].next = len(arr[sa_layerid][sid])-1
            #         else:
            #             for i in range(corenum, len(arr[sa_layerid][sid])):
            #                 if arr[sa_layerid][sid][i].coreid == cid and arr[sa_layerid][sid][i].next is None:
            #                     arr[sa_layerid][sid][i].next = len(arr[sa_layerid][sid])-1
            #                     break
            # else:
            #     arr[sa_layerid][sid][cid].coreid = cid
            #     arr[sa_layerid][sid][cid].centroid = centroid
            #     arr[sa_layerid][sid][cid].allocTime = allocTime
        elif 'is released' in line:
            sid_startpos = line.find('ce')+3
            sid = int(line[sid_startpos:sid_startpos+line[sid_startpos:].find(' ')])
            cid_startpos = sid_startpos+line[sid_startpos:].find('re')+3
            cid = int(line[cid_startpos:line.find('is')-1])
            releaseTime = int(line[line.rfind('le')+3:-1])

            arr[sa_layerid][sid][cid][-1].releaseTime = releaseTime
            arr[sa_layerid][sid][cid][-1].true_relTime = arr[sa_layerid][sid][cid][-1].releaseTime - arr[sa_layerid][sid][cid][-1].allocTime + arr[sa_layerid][sid][cid][-1].true_allocTime + mlp_latency[sa_layerid]

            # if arr[sa_layerid][sid][cid].releaseTime > 0:
            #     print('Attention - 2: lid: {}, sid: {}, cid: {}'.format(sa_layerid, sid, cid))
            #     for i in range(corenum, len(arr[sa_layerid][sid])):
            #         if arr[sa_layerid][sid][i].coreid == cid and arr[sa_layerid][sid][i].releaseTime == 0:
            #             arr[sa_layerid][sid][i].releaseTime = releaseTime
            #             break
            # else:
            #     arr[sa_layerid][sid][cid].releaseTime = releaseTime
    
    for lid in range(layernum):
        for sid in range(slicenum):
            # rtList = [arr[lid][sid][cid].releaseTime for cid in range(len(arr[lid][sid]))]
            # maxpos = rtList.index(max(rtList))
            for cid in range(corenum):
                for i in range(len(arr[lid][sid][cid])):
                    if arr[lid][sid][cid][i].releaseTime > 0:
                        print('{},{},{},{},{},{},{},{}'.format(lid, sid, cid, arr[lid][sid][cid][i].centroid, arr[lid][sid][cid][i].allocTime, arr[lid][sid][cid][i].true_allocTime, arr[lid][sid][cid][i].releaseTime, arr[lid][sid][cid][i].true_relTime))
                        # last_ret = ',last' if cid == maxpos else ''
                        # ret += last_ret
                        # print(ret)
    


