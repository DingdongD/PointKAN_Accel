import torch
# import dramAddrParse as mem_parser
import pdb
from print_log import OUTPUT
from dram import DataSrc

class SamplingEngine:
    def __init__(self, dc_nums, xyz, npoint):
        self.ndc = dc_nums
        N, C = xyz.shape
        self.pcd = xyz  # from global memory, for only 1 batch, namely xyz.shape = (N, 3)
        self.__allocate_ptr = 0 # a load pointer
        self.npoint = npoint
        self.__centroid_counter = 0 # count the number of centroids, to determine whether a batch is finished
        self.new_iter = True    # a flag to indicate whether a new iteration can be started

        self.local_buffer = torch.ones(N).cpu() * 1e10 # 'distance' in FPS implementation
        self.farthest = torch.randint(0, N, (1,), dtype=torch.long).cpu()[0].item()  # to global memory
        self.centroids = self.pcd[self.farthest, :] # from global memory
        
        # Timing parameters
        self.dc_latency = 0 # config parameters of the DC
        self.dc_latency_counters = torch.tensor(self.dc_latency).repeat(self.ndc).cpu()

        # Intermediate queues for data communication between components 
        self.dcs2min = torch.ones(self.ndc, 2).cpu() * -1 # (idx, dist) for each
        
        # Measure the traffic in and out of the Sampling Engine
        self.io_traffic = 3 # initially load 1 centroid's coordinates from the global memory

        # pending requests for responses from the DRAM
        self.pending_read_neighbors = []
        self.pending_read_centroids = []

        # response list, an interface bewteen the DRAM and the Sampling Engine
        self.dram_response = []

        # self.internal_clock = 0 # for debug
        self.is_first_centroid = True
        self.is_complete = False
        self.is_start = True
        
        self.clock = 0
    
    # def is_localbuf_full(self):
    #     return (self.local_buffer==1e10).nonzero().numel() == 0

    # return:   is_complete: a flag to indicate whether a batch is finished
    #           ret_centroid: the index of the farthest point (centroid idx)
    #           requests_list: a list of requests to the DRAM
    def reset(self, xyz, npoint):
        self.__allocate_ptr = 0 # a load pointer
        self.__centroid_counter = 0
        self.new_iter = True   
        self.pcd = xyz
        self.npoint = npoint
        self.local_buffer = torch.ones(self.pcd.shape[0]).cpu() * 1e10 
        self.farthest = torch.randint(0, self.pcd.shape[0], (1,), dtype=torch.long).cpu()[0]  # to global memory
        self.centroids = self.pcd[self.farthest, :] # from global memory
        self.dc_latency_counters = torch.tensor(self.dc_latency).repeat(self.ndc).cpu()
        self.dcs2min = torch.ones(self.ndc, 2).cpu() * -1 # (idx, dist) for each
        self.io_traffic = 3
        self.pending_read_neighbors = []
        self.pending_read_centroids = []
        self.dram_response = []
        # self.internal_clock = 0 # for debug
        self.is_first_centroid = True
        self.is_complete = False
        self.is_start = True
        self.clock = 0

    def cycle(self):
        ret_centroid = None
        requests_list = []

        # handling read responses from the DRAM
        read_response = [x[0] for x in self.dram_response if x[1] == False]
        for res in read_response:
            # ori_res = copy.deepcopy(res)
            # res = mem_parser.parse_addr2pidx(res)
            for i in range(len(self.pending_read_centroids)):
                if self.pending_read_centroids[i][0] == res and self.pending_read_centroids[i][1] > 0:
                    self.pending_read_centroids[i][1] -= 1
                    if self.pending_read_centroids[i][1] == 0:

                        self.centroids = self.pcd[res, :]
                        self.io_traffic += 3    # load a centroid's coordinates
                        self.new_iter = True
                        # print('[Read Response] receive centroid {} from DRAM'.format(res))
                        self.pending_read_centroids.remove([res, 0])
                        # self.dram_response.remove((ori_res, False))

                        # print('len(pending_read_centroids): {}, __centroid_counter={}, npoint[0]={}'.format(len(self.pending_read_centroids), self.__centroid_counter, self.npoint[0]))
                        # if self.pending_read_centroids == [] and self.__centroid_counter == self.npoint[0]:
                        #     print('A batch finished. io_traffic:', self.io_traffic*4/1024, 'KB')
                        #     self.local_buffer = torch.ones(self.pcd.shape[0]).cpu() * 1e10  # reset the local buffer
                        #     self.is_complete = True
                        #     self.new_iter = False
                    res = None
                    break
            
            if res is not None:
                for i in range(len(self.pending_read_neighbors)):
                    if self.pending_read_neighbors[i][0] == res and self.pending_read_neighbors[i][1] > 0:
                        self.pending_read_neighbors[i][1] -= 1
                        if self.pending_read_neighbors[i][1] == 0:
                            neighbor = self.pcd[res,:]
                            self.io_traffic += 3
            
                            centroid = self.centroids   
                            dist = torch.sum((neighbor - centroid) ** 2, -1)
                            for dc in range(self.ndc):
                                if self.dcs2min[dc, 0] == res and self.dcs2min[dc, 1] == -1:
                                    self.dcs2min[dc, 1] = dist
                                    break
                            self.pending_read_neighbors.remove([res, 0])
                            # self.dram_response.remove((ori_res, False))
                        break
            

        # MAX unit
        if self.is_first_centroid or (self.new_iter is False and self.pending_read_neighbors == [] and self.pending_read_centroids == [] and (self.dcs2min[:,1] != -1).nonzero().numel() == 0 and (self.local_buffer == 1e10).nonzero().numel() == 0 and self.is_complete is False):
            # exclude the 1st randomly generated centroid
            if not self.is_first_centroid:
                self.farthest = torch.max(self.local_buffer, -1)[1].item()    # argmax(dists)
            # addr_x, addr_y, addr_z = mem_parser.gen_xyzcoord_addr(self.farthest)
            # requests_list.append([addr_x, 0, 0])
            # requests_list.append([addr_y, 0, 0])
            # requests_list.append([addr_z, 0, 0])
            # print('[Read Request] send a read request to load the centroid {}\'s coordinates from DRAM'.format(self.farthest))
            requests_list.append((self.farthest, 12, DataSrc.fps_unit, False))
            self.pending_read_centroids.append([self.farthest, 3])
            ret_centroid = self.farthest.item() if isinstance(self.farthest, torch.Tensor) else self.farthest
            self.is_first_centroid = False
            self.__centroid_counter += 1
            # requests_list.append([mem_parser.gen_centroids_addr_withidx(ret_centroid), 1, 0])
            # print('[Write Request] send a write request to store the generted centroid {} to DRAM'.format(ret_centroid.item()))
            self.io_traffic += 1    # output a centroid

            if self.__centroid_counter == self.npoint and self.npoint is not None:
                # OUTPUT('A batch finished. io_traffic: {} KB, clock: {} cycles'.format(self.io_traffic*4/1024, self.clock+1))
                self.local_buffer = torch.ones(self.pcd.shape[0]).cpu() * 1e10  # reset the local buffer
                self.is_complete = True
                self.new_iter = False
                self.is_start = False
                self.pending_read_centroids = []
                requests_list = []
        
        # MIN unit
        for dc in range(self.ndc):
            if self.dcs2min[dc, 1] != -1:
                idx, dist = self.dcs2min[dc, :]
                idx = int(idx)
                self.local_buffer[idx] = torch.min(self.local_buffer[idx], dist)
                # print('[update local buffer] update local buffer[{}] = {}'.format(idx, self.local_buffer[idx]))
                self.dcs2min[dc, :] = torch.ones(2).cpu() * -1    # reset the dcs2min queue
                # print('[release dcs2min] release dcs2min queue[{}]'.format(dc))
        
        # Distance Calculator
        if self.new_iter:
            for dc in range(self.ndc):    
                if self.dcs2min[dc, 0] == -1:
                    if self.dc_latency_counters[dc] == 0:
                        if self.__allocate_ptr != self.farthest:
                            # addr_x, addr_y, addr_z = mem_parser.gen_xyzcoord_addr(self.__allocate_ptr)
                            # requests_list.append([addr_x, 0, 0])
                            # requests_list.append([addr_y, 0, 0])
                            # requests_list.append([addr_z, 0, 0])
                            requests_list.append((self.__allocate_ptr, 12, DataSrc.fps_unit, False))
                            self.pending_read_neighbors.append([self.__allocate_ptr, 3])
                            # print('[Read Request] send a read request to load the neighbor {}\'s coordinates from DRAM'.format(self.__allocate_ptr))
                        else:
                            self.dcs2min[dc, 1] = 0 # dist from current centroid to itself is 0

                        self.dcs2min[dc, 0] = self.__allocate_ptr

                        self.__allocate_ptr += 1
                        if self.__allocate_ptr == self.pcd.shape[0]:
                            self.__allocate_ptr = 0
                            self.new_iter = False

                        self.dc_latency_counters[dc] = self.dc_latency  # reset the counter
                    else:
                        self.dc_latency_counters[dc] -= 1
        
        if self.is_start is True:
            self.clock += 1
        return self.is_complete, ret_centroid, requests_list


def FPS_UnitTest():
    import data.test_data as data
    import dram
    
    xyz = data.xyz
    npoint = 8
    dc_nums = 1
    
    fps = SamplingEngine(dc_nums, xyz, [npoint])
    mem = dram.FixedLatencyDram()
    
    while True:
        is_complete, ret_centroid, requests_list = fps.cycle()
        if ret_centroid is not None:
            print('centroid:', ret_centroid)
        if is_complete:
            break
        mem.push(requests_list)
        fps.dram_response.extend(mem.cycle())


if __name__ == '__main__':
    FPS_UnitTest()