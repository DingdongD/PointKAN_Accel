import torch
import copy
# import dramAddrParse as mem_parser
import pdb
from dram import DataSrc
from print_log import OUTPUT

class NeighborSearchEngine:
    def __init__(self, K, sa_num, xyz, radius, npoints=0):
        self.neighbor_num = K   # number of neighbors for each point
        self.nsarr = sa_num     # architecture parameters, the number of sorter arrays
        self.ndc = sa_num       # architecture parameters, the number of distance calculators
        self.npoints = npoints
        self.radius2 = radius**2

        # global memory data
        self.xyz = xyz  # always record the original point cloud coordinates
        self.pcd = xyz
        self.load_pcd_ptrs_set = torch.arange(0, self.pcd.shape[0]).cpu()
        self.load_pcd_ptrs = torch.zeros(self.ndc).cpu()    # a group of pointers for reading neighbors in each DC
        self.centroids_pending_quque = []
        # centroid fixed in DCs' register, which is loaded from the global memory
        self.dcregs = torch.ones(self.ndc, 1+3).cpu() * -1  # (centroid) idx + coordinates

        # SEs of sorter arrays
        self.sorter_array = torch.ones(self.nsarr, self.neighbor_num, 2).cpu() * 1e10 # (idx, dist) for each

        # Intermediate queues for data communication between DCs and SAs
        self.dcs2sas = torch.ones(self.ndc, 2).cpu() * -1   # (neighbor) idx, dist
        
        # Timing
        self.dc_latency = 3
        self.dc_latency_counters = torch.tensor(self.dc_latency).repeat(self.ndc).cpu()

        # Measure the traffic in and out of the Neighbor Search Engine
        self.io_traffic = 0
        # Measure the maximum active sorter arrays in a cycle
        self.max_active_sas = 0
        # check whether all centroids have been processed, so that can terminal the simulation
        self.centroid_counter = 0

        # pending read requests
        self.pending_read_centroid = []
        self.pending_read_neighbor = []

        self.dram_response = []

        self.is_dc_busy = torch.zeros(self.ndc).cpu()

        # self.centroids = []
        self.clock = 0 # for debug
        self.neighbor_counter_per_dc = torch.zeros(self.ndc).cpu()
        self.is_idle = False

    def update_pcd(self, xyz, centroid_id, radius, npoint):
        self.pcd = xyz
        # self.load_pcd_ptrs_set = [int(x) for x in centroid_id]
        self.load_pcd_ptrs_set = torch.arange(0, self.pcd.shape[0]).cpu()
        self.radius2 = radius**2
        self.npoints = npoint
        self.is_idle = False
        self.clock = 0
        # OUTPUT('[update pcd] pcd.shape={}, xyz={}'.format(self.pcd.shape, xyz))
    
    def cycle(self, centroid=None):
        ret = []
        requests_list = []
        active_sas = 0

        # Receive a centroid and append it to the pending queue
        if centroid is not None:
            self.centroids_pending_quque.append(centroid)   # Note: centroid (id) is an element, not list or tensor
        
        # handling dram response
        read_response = [x[0] for x in self.dram_response if x[1] == False]
        for res in read_response:
            # res = mem_parser.parse_addr2pidx(res)
            for i in range(len(self.pending_read_centroid)):
                if self.pending_read_centroid[i][0] == res and self.pending_read_centroid[i][1] > 0:
                    self.pending_read_centroid[i][1] -= 1
                    if self.pending_read_centroid[i][1] == 0:
                        for dc in range(self.ndc):
                            if self.dcregs[dc, 0] == res and self.dcregs[dc, 1:].sum() == -3:
                                # OUTPUT('[Read Response] receive a read response for centroid {} in DC {}, clock: {}'.format(res, dc, self.clock))
                                # self.dcregs[dc, 1:] = self.pcd[res]
                                self.dcregs[dc, 1:] = self.xyz[res]
                                self.load_pcd_ptrs[dc] = 0

                                self.io_traffic += 4 
                                self.pending_read_centroid.remove([res, 0])
                                break
                    res = None
                    break
            
            if res is not None:
                for i in range(len(self.pending_read_neighbor)):
                    if self.pending_read_neighbor[i][0] == res and self.pending_read_neighbor[i][1] > 0:
                        self.pending_read_neighbor[i][1] -= 1
                        if self.pending_read_neighbor[i][1] == 0:
                            neighbor = self.xyz[res,:]
                            for dc in range(self.ndc):
                                if self.dcs2sas[dc, 0] == res and self.dcs2sas[dc, 1] == -1:
                                    # OUTPUT('[Read Response] receive a read response for neighbor {} in DC {}, clock: {}'.format(res, dc, self.clock))
                                    centroid = self.dcregs[dc, 1:]
                                    if (centroid==-1).nonzero().numel() == 3:
                                        # centroid coordinates are not loaded yet
                                        OUTPUT('centroid coordinates are not loaded yet')
                                        break
                                    dist = torch.sum((neighbor - centroid) ** 2, -1)
                                    if dist > self.radius2:
                                        self.dcs2sas[dc, 0] = -1
                                    else:
                                        self.dcs2sas[dc, 1] = dist
                                    self.neighbor_counter_per_dc[dc] += 1

                                    self.io_traffic += 3    # load a neighbor's coordinates
                                    self.pending_read_neighbor.remove([res, 0])
                                    # self.dram_response.remove((ori_res, False))
                                    break                
                        break
        
        # Sorter Array
        for sa in range(self.nsarr):
            if self.dcs2sas[sa, 1] >= 0:    # namely dist(centroid, neighbor) >= 0
                comp_tmp = (self.dcs2sas[sa, 1] < self.sorter_array[sa, :, 1]).nonzero()
                # self.neighbor_counter_per_dc[sa] += 1
                if comp_tmp.numel() > 0:
                    pos = int(comp_tmp[0])
                    self.sorter_array[sa] = torch.cat((self.sorter_array[sa, :pos], self.dcs2sas[sa].view(1,2), self.sorter_array[sa, pos:self.neighbor_num-1]), dim=0) # insert the newly come neighbor to the right position (i.e. pos) in the sorter array
                self.dcs2sas[sa, :] = torch.ones(2, dtype=torch.long).cpu() * -1 # reset the queue
            
            if self.load_pcd_ptrs[sa] == self.pcd.shape[0] and self.neighbor_counter_per_dc[sa] == self.pcd.shape[0]:
                self.centroid_counter += 1  # a centroids complete its neighbor search task
                self.is_dc_busy[sa] = 0
                self.neighbor_counter_per_dc[sa] = 0
                # OUTPUT('[NS-Release] DC: {} completes centroid {}\'s neighbor search, clock: {}'.format(sa, int(self.dcregs[sa, 0]), self.clock))
                self.io_traffic += self.neighbor_num
                # addr0, addr1, addr2, addr3 = mem_parser.gen_neighbors_addr_withidx(self.centroid_counter)
                # requests_list.append([addr0, 1, 1])
                # requests_list.append([addr1, 1, 1])
                # requests_list.append([addr2, 1, 1])
                # requests_list.append([addr3, 1, 1])
                # OUTPUT('[Send Request] SA {} sends a write request for centroid {}\'s neighbors, clock {}'.format(sa, int(self.dcregs[sa, 0]), self.clock))
                self.sorter_array[sa, (self.sorter_array[sa, :, 0] == 1e10).nonzero(), 0] = self.sorter_array[sa, 0, 0]
                ret.append(copy.deepcopy(self.sorter_array[sa, :, 0].type(torch.int32).tolist()))
                self.sorter_array[sa] = torch.ones(self.neighbor_num, 2).cpu() * 1e10 # reset the queue
                self.dcregs[sa] = torch.ones(1+3).cpu() * -1 # reset the DC register
                if self.npoints > 0 and self.centroid_counter == self.npoints:
                    # OUTPUT('[NS-Exit] NS have been completed, io traffic: {} MB, max active sas: {}, clock: {}'.format(self.io_traffic*4/1024/1024, self.max_active_sas, self.clock))
                    self.centroid_counter = 0
                    self.is_idle = True

        # Distance Calculator 
        for dc in range(self.ndc):
            # Insure the DC register is not empty, which means the current DC is allocated.
            if torch.sum(self.dcregs[dc, 1:]).item() != -3 and self.dcs2sas[dc, 0] == -1 and self.load_pcd_ptrs[dc] < self.pcd.shape[0]:    # at least the request for centroid coordinates has been sent
                active_sas += 1
                if self.dc_latency_counters[dc] == 0:
                    # addr_x, addr_y, addr_z = mem_parser.gen_xyzcoord_addr(int(self.load_pcd_ptrs_set[int(self.load_pcd_ptrs[dc])]))
                    # requests_list.append([addr_x, 0, 1])
                    # requests_list.append([addr_y, 0, 1])
                    # requests_list.append([addr_z, 0, 1])
                    requests_list.append((int(self.load_pcd_ptrs_set[int(self.load_pcd_ptrs[dc])]), 12, DataSrc.knn_unit, False))
                    self.pending_read_neighbor.append([int(self.load_pcd_ptrs_set[int(self.load_pcd_ptrs[dc])]), 3])
                    # OUTPUT('[Send Request] DC {} sends a read request for neighbor {}, clock: {}'.format(dc, int(self.load_pcd_ptrs_set[int(self.load_pcd_ptrs[dc])]), self.clock))
                    
                    self.dcs2sas[dc, 0] = int(self.load_pcd_ptrs_set[int(self.load_pcd_ptrs[dc])])
                    self.load_pcd_ptrs[dc] += 1                    
                    self.dc_latency_counters[dc] = self.dc_latency  # reset the counter
                else:
                    self.dc_latency_counters[dc] -= 1

            # allocate a centroid to a DC
            if (self.dcregs[dc] == -1).nonzero().numel() == 4 and len(self.centroids_pending_quque) > 0 and self.is_dc_busy[dc] == 0:
                centroid_id = self.centroids_pending_quque.pop(0)
                # OUTPUT('[NS-Allocation] allocate centroid {} to DC {} at cycle {}'.format(centroid_id, dc, self.clock))
                self.is_dc_busy[dc] = 1
                # requests_list.append([centroid_id, 0, 1])  # load a centroid (include its index and coordinates) from the global memory
                # addr_x, addr_y, addr_z = mem_parser.gen_xyzcoord_addr(centroid_id)
                # requests_list.append([addr_x, 0, 1])
                # requests_list.append([addr_y, 0, 1])
                # requests_list.append([addr_z, 0, 1])
                requests_list.append((centroid_id, 12, DataSrc.knn_unit, False))
                self.pending_read_centroid.append([centroid_id, 3])
                self.dcregs[dc, 0] = centroid_id
                # OUTPUT('[Send Request] DC {} sends a read request for centroid {}, clock: {}'.format(dc, centroid_id, self.clock))

        self.max_active_sas = max(active_sas, self.max_active_sas)
        
        self.clock += 1
        return self.is_idle, ret, requests_list
    
def KNN_UnitTest(mode='large'):
    
    if mode == 'large':
        import data.test_data as cen_data
        import data.data as xyz_data
        centroid = cen_data.centroids
        xyz = xyz_data.xyz_1024
        nsample = 32
    else:   # tiny
        import data.test_data as data
        xyz = data.xyz
        centroid = data.centroids_demo
        nsample = 2
    
    import dram
    mem = dram.FixedLatencyDram()
    response = []
    
    neigh_cnt = 0
    
    knn = NeighborSearchEngine(nsample, 1, xyz, 0.2, [len(centroid)])
    while True:
        if len(centroid) > 0:
            cid = centroid.pop(0)
        else:
            cid = None
        state, neighbors, rqst = knn.cycle(cid)
        
        if len(rqst) > 0:
            mem.push(rqst)
            # OUTPUT('[request] ', rqst)
        
        if neighbors is not None and len(neighbors) > 0:
            neigh_cnt += len(neighbors)
            
            if state is True:
                print('COMPLETE, clock:', knn.clock)
                break
        
        response = mem.cycle()
        knn.dram_response.extend(response)


if __name__ == '__main__':
    KNN_UnitTest()