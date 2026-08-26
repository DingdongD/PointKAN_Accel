import libdramsim3 as dram
import baseline_sampling_engine as spl
# import sampling_engine as spl
# import clustering_engine as clu
import neighbor_search_engine as ns
import baseline_feature_engine as fc

import torch
import pdb
import numpy as np
import data
import dramAddrParse as parser

class TabArchSys:
    # parse config parameters
    def __init__(self, xyz, npoints, nsK, radius):
        self.clk = 0    # system clock

        # global shared parameters (pointnet2 native)
        self.pcd = xyz          # point cloud data
        self.npoints = npoints    # number of centroids 
        self.nsK = nsK          # number of neighbors searched for each centroid
        assert self.pcd.shape[0] >= self.nsK, 'The scale of point cloud is too small, does not match the number of neighbors searched for each centroid (i.e. K in KNN).'
        self.radius = radius    # radius of neighbor search

        # local parameters
        self.spl_ndc = 16       # number of distance calculators in sampling engine
        self.ns_nsa = 16        # number of sorter arrays in neighbor search engine
        # self.clu_ncenter = 8   # number of centers in clustering engine 
        self.fc_nslice = 8      # number of slices in feature computing engine
        self.fc_ncore = 16       # number of cores in each slice of feature computing engine
        self.fc_szhashtab = 640   # size of hash table in each slice of feature computing engine, 2688
        
        # components
        self.sampling_engine = spl.SamplingEngine(self.spl_ndc)
        # self.sampling_engine = spl.SamplingEngine(self.spl_ndc, self.pcd, self.npoints)
        self.neighbor_search_engine = ns.NeighborSearchEngine(self.nsK, self.ns_nsa, self.pcd, self.radius[0], self.npoints)
        # self.clustering_engine = clu.ClusteringEngine(self.clu_ncenter, self.pcd)   # lacks centroids set
        self.feature_computing_engine = fc.FeatureComputingEngine(self.fc_nslice, self.fc_ncore, self.fc_szhashtab, self.pcd)

        # collect requests from each component
        self.requests = []
        self.requests_bak = []

        # centroids from sampling engine
        self.centroids = [] 
        self.clusters = []
        self.ns_send_centroids_ptr = 0
        self.ns_receive_centroids_ptr = 0
        self.neighbors = []
        for i in range(len(self.npoints)):
            if self.npoints[i] is not None:
                self.neighbors.append([])
            else:
                self.neighbors.append(None)

        self.sa_layerid = 0    # record the index of SA layer
        self.total_sa_layer = len(self.npoints)
        self.clu_npoints_ptr = 0
        self.ns_npoints_ptr = 0
        self.ban_spl = False

        self.fixed_dram = Dram()
        self.global_buffer = []
        self.redreq_by_globalbuf = 0
        self.totreq = 0

        # temp variables for debug
        # self.sa_layerid = 0 # for debug
        # self.total_sa_layer = 1 # for debug
        # # self.ns_send_centroids_ptr = 1 # for debug
        # self.pcd = torch.index_select(self.pcd, 0, torch.tensor(data.centroids_512))
        # self.ns_send_centroids_ptr = 0
        # self.neighbor_search_engine.npoint = self.npoints[self.sa_layerid]
        # self.neighbor_search_engine.update_pcd(self.pcd, data.centroids_512, self.radius[self.sa_layerid])
        # # self.centroids = self.centroids[1]
        # self.centroids = data.centroids_128 # for debug\
        # self.neighbors = data.neighbors_128 # for debug
        
    
    def distribute_dram_response(self, dram_response):
        self.sampling_engine.dram_response.extend([x[0:2] for x in dram_response if x[2] == 0])
        self.neighbor_search_engine.dram_response.extend([x[0:2] for x in dram_response if x[2] == 1])
        # self.clustering_engine.dram_response.extend([x[0:2] for x in dram_response if x[2] == 2])
        self.feature_computing_engine.dram_response.extend([x[0:2] for x in dram_response if x[2] == 3])

    def cycle(self):
        # dramsim
        # dram_response = dram.cycle()
        # self.distribute_dram_response(dram_response)

        # feature computing engine simulation
        if self.npoints[self.sa_layerid] is not None and len(self.centroids) == self.npoints[self.sa_layerid]:
            centroids_requests = self.feature_computing_engine.input_centroids(self.centroids)
            self.requests.extend(centroids_requests)
        if self.npoints[self.sa_layerid] is not None and len(self.neighbors[self.sa_layerid]) == self.npoints[self.sa_layerid]:
            neighbors_requests = self.feature_computing_engine.input_neighbors(self.neighbors[self.ns_npoints_ptr])
            self.requests.extend(neighbors_requests)
            
        fc_state, fc_requests = self.feature_computing_engine.cycle()
        if len(fc_requests) > 0:
            self.requests.extend(fc_requests)
            self.requests_bak.extend(fc_requests)   # for deubg
        if fc_state == -1 and self.sa_layerid < self.total_sa_layer:
            print('SA layer {} finished, time consuming: {}'.format(self.sa_layerid, self.clk))
            self.sa_layerid += 1
            self.requests = [x for x in self.requests if x[2] != 3]
            # reset sampling
            self.pcd = torch.index_select(self.pcd, 0, torch.tensor(self.centroids))
            print('update xyz.shape={}, layerid: {}'.format(self.pcd.shape,self.sa_layerid))
            self.ban_spl = False if self.sa_layerid < 3 else True
            # reset neighbor search
            self.ns_send_centroids_ptr = 0
            self.neighbor_search_engine.npoint = self.npoints[self.sa_layerid]
            self.neighbor_search_engine.update_pcd(self.pcd, self.centroids, self.radius[self.sa_layerid])
            self.centroids = []
            
        if fc_state == 1:
            print('[System Report] Feature Computing finished, time consuming: ', self.clk)
            fc_state = True


        # neighbor search engine simulation
        centroid = None
        if len(self.centroids) == self.npoints[self.sa_layerid] and self.ns_send_centroids_ptr < self.npoints[self.sa_layerid]:
            centroid = self.centroids[self.ns_send_centroids_ptr]
            self.ns_send_centroids_ptr += 1
        ns_state, ns_neighbors, ns_requests = self.neighbor_search_engine.cycle(centroid)
        if len(ns_requests) > 0:
            self.requests.extend(ns_requests)
        if len(ns_neighbors) > 0:
            ns_neighbors = [[int(x) for x in tensor] for tensor in ns_neighbors]
            self.neighbors[self.sa_layerid].extend(ns_neighbors)
            # print('[Single Output NS] neighbors:', ns_neighbors, ', clock: ', self.clk)
            if len(self.neighbors[self.sa_layerid]) % 32 == 0:
                    print('neighbor search progress: {}/{}, timer: {}'.format(len(self.neighbors[self.sa_layerid]), self.npoints[self.sa_layerid], self.clk))
            if ns_state and len(self.neighbors[self.sa_layerid]) == self.npoints[self.sa_layerid]:
                print('[System Report] Neighbor Search finished, size =', len(self.neighbors[self.sa_layerid]),', time consuming: ', self.clk)
                print('[Output-NS] neighbors:', self.neighbors[self.sa_layerid], ', clock: ', self.clk)
        if ns_state is True:
            self.requests = [x for x in self.requests if x[2] != 1]
        ns_state = ns_state and self.sa_layerid == len(self.npoints)-1

        # sampling engine simulation
        if not self.ban_spl:
            spl_state, spl_centroid, spl_requests = self.sampling_engine.cycle()
            
            if len(spl_requests) > 0:
                self.requests.extend(spl_requests)
            if spl_centroid is not None:
                print('[Output-SPL] centroid:', spl_centroid.item(), ', clock: ', self.clk)
                self.centroids.append(spl_centroid.item())
            if spl_state is True:
                self.requests = [x for x in self.requests if x[2] != 0]
                if len(self.centroids) == self.npoints[self.sa_layerid]:
                    print('[System Report] Sampling (FPS) finished, centroids.size =',len(self.centroids),', centroids:', self.centroids,', time consuming:', self.clk, 'cycles')
                    self.ban_spl = True
                    
            if self.sampling_engine.pcd is None and self.ban_spl is False and self.sa_layerid < len(self.npoints):    # TODO: update the if condiction
                print('[SPL-Input] self.pcd.shape: {}, self.npoints[{}]: {}'.format(self.pcd.shape, self.sa_layerid, self.npoints[self.sa_layerid]))
                self.sampling_engine.input(self.pcd, self.npoints[self.sa_layerid])

        # # send requests to dram
        # if len(self.requests) > 0:
        #     req = self.requests.pop(0)
        #     dram_response = self.fixed_dram.push([int(req[0], 16), False if req[1]==0 else True, req[2]])
        
        # dram_response = self.fixed_dram.cycle()
        # if len(dram_response) > 0:
        #     self.distribute_dram_response([dram_response])
        ptr = 0
        while len(self.requests) > 0 and ptr < len(self.requests):
            req = self.requests[ptr]
            addr = int(req[0], 16)
            typ = parser.dtype(addr)
            is_succeed = False
            if typ != 4:    # exclude feature vectors
                if addr in self.global_buffer:
                    self.distribute_dram_response([[addr, True if req[1]==1 else False, req[2]]])
                    is_succeed = True
                    self.redreq_by_globalbuf += 1

            if is_succeed is False:
                # is_succeed = dram.add_pending(addr, True if req[1]==1 else False, req[2])  
                self.fixed_dram.push([int(req[0], 16), False if req[1]==0 else True, req[2]])
                is_succeed = True
                
            if is_succeed is True:
                self.requests.pop(ptr)
                self.totreq += 1
                break
            else:
                ptr += 1
                
        dram_response = self.fixed_dram.cycle()
        if len(dram_response) > 0 and dram_response[1] == False:
            readcallback_addrs = dram_response[0]
            typ = parser.dtype(readcallback_addrs)

            if typ != 4 and readcallback_addrs not in self.global_buffer:
                self.global_buffer.append(readcallback_addrs)
            self.distribute_dram_response([dram_response])
            # self.requests = [x for x in self.requests if x[1] == False and x[0] != readcallback_addrs]
                

        self.clk += 1

        return self.ban_spl and ns_state and fc_state
    
    def SysUnitTest(self):
        while True:
            state = self.cycle()
            if state is True:
                print('Completed!!!, global_buffer size:', len(self.global_buffer), ', access reduced by global buffer:', self.redreq_by_globalbuf/self.totreq*100,'%')
                break

class Dram:

    class DramEntry:
        def __init__(self, request):
            self.request = request
            self.timer = 50

    def __init__(self):
        self.pending_list = []
    
    def cycle(self):            
        return_list = None
        for i in range(len(self.pending_list)):
            self.pending_list[i].timer -= 1
            if self.pending_list[i].timer <= 0:
                return_list = self.pending_list[i].request
            
        # pop the request from the pending list when its timer is 0
        self.pending_list = list(filter(lambda req:req.timer>0, self.pending_list))

        return return_list if return_list is not None else []

    def push(self, requests):
        if len(requests) > 0:
            # for req in requests:
            self.pending_list.append(self.DramEntry(requests))

    def debug_show_dram(self):
        output = {}
        for req in self.pending_list:
            output[req.request] = req.timer
        print(output)


if __name__ == "__main__":
    import time
    start = time.time()
    system = TabArchSys(data.xyz_1024, [512,128,128], 32, [0.2,0.4,0.4])
    # system = TabArchSys(data.xyz_1024, [512, 128], 32, [0.2, 0.4])
    # system = TabArchSys(data.xyz_128, [64,16,16], 4)
    # system = TabArchSys(data.xyz_16, [8,4,4], 2, [0.2,0.4,0.4])
    system.SysUnitTest()
    stop = time.time()
    print('Time consuming: ', stop-start, 's')