import libdramsim3 as dram
import sampling_engine as spl
import clustering_engine as clu
import neighbor_search_engine as ns
import feature_computing_engine as fc
import dramAddrParse as parser

import torch
import pdb
import numpy as np
import data
import s3dis
import shapenet

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

        # local parameters (modelnet40 for cls)
        self.spl_ndc = 16        # number of distance calculators in sampling engine
        self.ns_nsa = 16        # number of sorter arrays in neighbor search engine
        self.clu_ncenter = 8    # number of centers in clustering engine 
        self.fc_nslice = 1      # number of slices in feature computing engine, 8
        self.fc_ncore = 16      # number of cores in each slice of feature computing engine
        self.fc_szhashtab = 128*(10+16+1)   # size of hash table in each slice of feature computing engine, 2688

        # configuration (s3dis for semseg)
        # self.spl_ndc = 2        # number of distance calculators in sampling engine
        # self.ns_nsa = 2        # number of sorter arrays in neighbor search engine
        # self.clu_ncenter = 4    # number of centers in clustering engine 
        # self.fc_nslice = 4      # number of slices in feature computing engine
        # self.fc_ncore = 16      # number of cores in each slice of feature computing engine
        # self.fc_szhashtab = 128*(10+16+1)   # size of hash table in each slice of feature computing engine, 2688

        # configuration (shapenet for part seg)
        # self.spl_ndc = 2        # number of distance calculators in sampling engine
        # self.ns_nsa = 2        # number of sorter arrays in neighbor search engine
        # self.clu_ncenter = 4    # number of centers in clustering engine 
        # self.fc_nslice = 4      # number of slices in feature computing engine
        # self.fc_ncore = 16      # number of cores in each slice of feature computing engine
        # self.fc_szhashtab = 128*(10+16+1)   # size of hash table in each slice of feature computing engine, 2688
        
        # components
        self.sampling_engine = spl.SamplingEngine(self.spl_ndc, self.pcd, self.npoints)
        self.neighbor_search_engine = ns.NeighborSearchEngine(self.nsK, self.ns_nsa, self.pcd, self.radius[0], self.npoints)
        self.clustering_engine = clu.ClusteringEngine(self.clu_ncenter, self.pcd)   # lacks centroids set
        self.feature_computing_engine = fc.FeatureComputingEngine(self.fc_nslice, self.fc_ncore, self.fc_szhashtab, self.pcd)

        # collect requests from each component
        self.requests = []

        # centroids from sampling engine
        self.centroids = [] 
        self.clusters = []
        self.ns_send_centroids_ptr = 0
        self.ns_receive_centroids_ptr = 0
        self.neighbors = []
        for i in range(len(self.npoints)):
            if self.npoints[i] is not None:
                # self.neighbors.append(torch.ones(self.npoints[i], self.nsK).cpu()*-1)
                self.neighbors.append([])
            else:
                self.neighbors.append(None)

        self.sa_layerid = 0    # record the index of SA layer
        self.total_sa_layer = len(self.npoints)
        # self.clu_layer_ptr = 0
        self.clu_npoints_ptr = 0
        self.ns_npoints_ptr = 0

        # temp variables for debug
        # self.total_sa_layer = 1  # for debug
        # self.centroids = 
        self.ban_spl = False    # normal execution
        # self.centroids = self.centroids[:128]   # for debug
        # self.ban_spl = True # for debug
        self.global_buffer = []
        self.redreq_by_globalbuf = 0
        self.totreq = 0
        self.is_feed_data_to_fce = True

        self.fixed_dram = Dram()
    
    def distribute_dram_response(self, dram_response):
        self.sampling_engine.dram_response.extend([list(x[0:2]) for x in dram_response if x[2] == 0])
        self.neighbor_search_engine.dram_response.extend([list(x[0:2]) for x in dram_response if x[2] == 1])
        self.clustering_engine.dram_response.extend([list(x[0:2]) for x in dram_response if x[2] == 2])
        self.feature_computing_engine.dram_response.extend([list(x[0:2]) for x in dram_response if x[2] == 3])

    def cycle(self):
        # dramsim
        # dram_response = dram.cycle()
        # readcallback_addrs = [x[0] for x in dram_response if x[1] == 0]
        # self.global_buffer.extend([x for x in readcallback_addrs if x not in self.global_buffer])
        # self.distribute_dram_response(dram_response)
        # readrequests = [x[0] for x in self.requests if x[1] == False]
        # for x in readcallback_addrs:
        #     x = hex(x)
        #     while x in readrequests:
        #         idx = readrequests.index(x)
        #         self.distribute_dram_response([(int(x, 16), False, self.requests[idx][2])])
        #         self.requests.pop(idx)
        #         readrequests.pop(idx)
        
        # feature computing engine simulation
        if self.is_feed_data_to_fce is True and self.sa_layerid < len(self.npoints):
            if self.npoints[self.sa_layerid] is None:
                last_non_none_layer = self.sa_layerid - 1
                while last_non_none_layer >= 0 and self.npoints[last_non_none_layer] is None:
                    last_non_none_layer -= 1
                self.feature_computing_engine.input_xyz(self.centroids[:self.npoints[self.sa_layerid-1]])
                self.feature_computing_engine.set_group_all_label(True)
                self.is_feed_data_to_fce = False
                # self.requests.extend(centroids_requests)
            else:
            # if self.npoints[0] > 0 and len(self.centroids) == self.npoints[0]:
                # if self.sa_layerid < len(self.npoints) and self.npoints[self.sa_layerid] is not None and len(self.centroids) >= self.npoints[self.sa_layerid]:
                #     centroids_requests = self.feature_computing_engine.input_centroids(self.centroids)
                #     self.requests.extend(centroids_requests)
                if self.sa_layerid < len(self.npoints) and self.npoints[self.sa_layerid] is not None and len(self.neighbors[self.sa_layerid]) == self.npoints[self.sa_layerid]:
                    neighbors_requests = self.feature_computing_engine.input_neighbors(self.neighbors[self.sa_layerid])
                    self.requests.extend(neighbors_requests)
                if len(self.clusters) > self.sa_layerid:    # self.feature_computing_engine.clusters is None and 
                    clusters_requests = self.feature_computing_engine.input_clusters(self.clusters[self.sa_layerid])
                    self.requests.extend(clusters_requests)
            
                if self.feature_computing_engine.neighbors is not None and self.feature_computing_engine.clusters is not None:
                    self.feature_computing_engine.set_group_all_label(False)
                    self.is_feed_data_to_fce = False
            
        fc_state, fc_requests = self.feature_computing_engine.cycle()
        if len(fc_requests) > 0:
            self.requests.extend(fc_requests)
        if fc_state == -1 and self.sa_layerid < self.total_sa_layer:
            print('SA {} finished at {}'.format(self.sa_layerid, self.clk))
            self.sa_layerid += 1
            self.is_feed_data_to_fce = True
            
            # if self.sa_layerid < len(self.npoints) and self.npoints[self.sa_layerid] is None:
            #     self.is_feed_data_to_fce = False
                # self.feature_computing_engine.set_group_all_label(True)
            self.requests = [x for x in self.requests if x[2] != 3]
            # centroids_requests = self.feature_computing_engine.input_centroids(self.centroids[:self.npoints[self.sa_layerid]])
            # self.requests.extend(centroids_requests)
            # if self.neighbors[self.sa_layerid] is not None:
            #     neighbors_requests = self.feature_computing_engine.input_neighbors(self.neighbors[self.sa_layerid])
            #     self.requests.extend(neighbors_requests)
            # if self.feature_computing_engine.clusters is None and len(self.clusters) > self.sa_layerid:
            #     clusters_requests = self.feature_computing_engine.input_clusters(self.clusters[self.total_sa_layer-self.sa_layerid-1])
            #     self.requests.extend(clusters_requests)
        if fc_state == 1:
            print('SA {} finished at {}'.format(self.sa_layerid, self.clk))
            print('[System Report] Feature Computing finished, time consuming: ', self.clk)
            fc_state = True

        # clustering engine simulation
        # if self.clu_npoints_ptr < self.total_sa_layer and len(self.centroids) >= self.npoints[self.total_sa_layer-self.clu_npoints_ptr-1] and self.clustering_engine.centroids is None:
        #     self.clustering_engine.input(self.centroids[:self.npoints[self.total_sa_layer-self.clu_npoints_ptr-1]])
        while self.clu_npoints_ptr < len(self.npoints) and self.npoints[self.clu_npoints_ptr] is None:
            self.clu_npoints_ptr += 1
        if self.clu_npoints_ptr < len(self.npoints) and len(self.centroids) >= self.npoints[self.clu_npoints_ptr] and self.clustering_engine.centroids is None:
            self.clustering_engine.input(self.centroids[:self.npoints[self.clu_npoints_ptr]])
        clu_state, clu_results, clu_requests = self.clustering_engine.cycle()
        if len(clu_requests) > 0: 
            self.requests.extend(clu_requests)
        if clu_results is not None:
            print('[clustering] result:', clu_results)
            self.clu_npoints_ptr += 1
            self.clusters.append(clu_results)
        
            # if len(self.clusters) == self.total_sa_layer:
            # if self.feature_computing_engine.clusters is None:
            #     clusters_requests = self.feature_computing_engine.input_clusters(self.clusters[-1])
            #     self.requests.extend(clusters_requests)
            # if clu_state:
            print('[System Report] Clustring finished, clusters.size =',len([item for sublist in clu_results for item in sublist]),', time consuming: ', self.clk)
            print('Memory reduction contributes by the global buffer: ', self.redreq_by_globalbuf/self.totreq*100, '%')
            clu_state = clu_state and self.clu_npoints_ptr == self.total_sa_layer
            if clu_state:
                self.requests = [x for x in self.requests if x[2] != 2]

        # neighbor search engine simulation
        centroid = None
        if self.ns_send_centroids_ptr < len(self.centroids) and self.ns_npoints_ptr < len(self.npoints) and self.npoints[self.ns_npoints_ptr] is not None and self.ns_send_centroids_ptr < self.npoints[self.ns_npoints_ptr]:
            centroid = self.centroids[self.ns_send_centroids_ptr]
            self.ns_send_centroids_ptr += 1
        ns_state, ns_neighbors, ns_requests = self.neighbor_search_engine.cycle(centroid)
        if len(ns_requests) > 0:
            self.requests.extend(ns_requests)

        if ns_neighbors is not None and len(ns_neighbors) > 0:
            while len(ns_neighbors) > 0:
                tmp = ns_neighbors.pop(0).tolist()
                self.neighbors[self.ns_npoints_ptr].append([int(x) for x in tmp]) # TODO
                if len(self.neighbors[self.ns_npoints_ptr]) % 32 == 0:
                    print('neighbor search progress: {}/{}, timer: {}'.format(len(self.neighbors[self.ns_npoints_ptr]), self.npoints[self.ns_npoints_ptr], self.clk))
        
            if len(self.neighbors[self.ns_npoints_ptr]) == self.npoints[self.ns_npoints_ptr]:
                # neighbors_requests = self.feature_computing_engine.input_neighbors(self.neighbors[self.ns_npoints_ptr])
                # self.requests.extend(neighbors_requests)
                if ns_state:
                    print('[System Report] Neighbor Search finished, size =', len(self.neighbors[self.ns_npoints_ptr]),', time consuming: ', self.clk)
                    print('Memory reduction contributes by the global buffer: ', self.redreq_by_globalbuf/self.totreq*100, '%')
                    print('neighbors:',self.neighbors[self.ns_npoints_ptr])
                
                self.ns_npoints_ptr += 1
                if self.ns_npoints_ptr < len(self.npoints) and self.npoints[self.ns_npoints_ptr] is not None:
                    # begin the next sa layer's neighbor search
                    print('[NS-Engine] Begin the next sa layer\'s (i.e. layer {}) neighbor search'.format(self.ns_npoints_ptr))
                    self.ns_send_centroids_ptr = 0
                    self.ns_receive_centroids_ptr = 0
                    self.neighbor_search_engine.update_pcd(torch.index_select(self.pcd, 0, torch.tensor(self.centroids[:self.npoints[self.ns_npoints_ptr-1]])), self.centroids[:self.npoints[self.ns_npoints_ptr-1]], self.radius[self.ns_npoints_ptr-1])
                    # TODO: update xyz to self.pcd[:self.npoints[self.ns_npoints_ptr-1]]
                else:
                    self.ns_npoints_ptr += 1

        ns_state = ns_state and self.ns_npoints_ptr >= len(self.npoints)
        if ns_state is True:
            self.requests = [x for x in self.requests if x[2] != 1]


        # sampling engine simulation
        if not self.ban_spl:
            spl_state, spl_centroid, spl_requests = self.sampling_engine.cycle()
            if len(spl_requests) > 0:
                self.requests.extend(spl_requests)
            if spl_centroid is not None:
                print('[Output-SPL] centroid:', spl_centroid.item(), ', clock: ', self.clk)
                # if self.totreq > 0:
                #     print('Access reduced by global buffer:', self.redreq_by_globalbuf/self.totreq*100,'%')
                self.centroids.append(spl_centroid)
            if spl_state is True:
                print('[System Report] Sampling (FPS) finished, centroids.size =',len(self.centroids),', centroids:', [x.item() for x in self.centroids], ', time consuming:', self.clk, 'cycles')
                print('Memory reduction contributes by the global buffer: ', self.redreq_by_globalbuf/self.totreq*100, '%')
                self.ban_spl = True
                self.requests = [x for x in self.requests if x[2] != 0]
                # pdb.set_trace()

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
        
        # # send requests to dram (for debug version)
        # while len(self.requests) > 0:
        #     req = self.requests[0]
        #     self.distribute_dram_response([(int(req[0], 16), True if req[1]==1 else False, req[2])])
        #     self.requests.pop(0)
                
        dram_response = self.fixed_dram.cycle()
        if len(dram_response) > 0 and dram_response[1] == False:
            readcallback_addrs = dram_response[0]
            typ = parser.dtype(readcallback_addrs)

            if typ != 4 and readcallback_addrs not in self.global_buffer:
                self.global_buffer.append(readcallback_addrs)
            self.distribute_dram_response([dram_response])
            self.requests = [x for x in self.requests if x[1] == False and x[0] != readcallback_addrs]

        self.clk += 1

        return self.ban_spl and ns_state and clu_state and fc_state
    
    def SysUnitTest(self):
        while True:
            state = self.cycle()
            if state is True:
                print('Complete!!!')
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
            output[req.request[0]] = req.timer
        print(output)

if __name__ == "__main__":
    # system = TabArchSys(data.xyz_1024, [512, 128, None], 32, [0.2, 0.4, None])
    # system = TabArchSys(data.xyz_1024, [512, 128], 32, [0.2, 0.4])
    # system = TabArchSys(data.xyz_128, [64,16,16], 4, [0.2, 0.4, 0.4])
    # system = TabArchSys(data.xyz_4, [2,2,None], 2, [0.4,0.4,None])
    system = TabArchSys(data.xyz_16, [4,2,None], 2, [0.4,0.8,None])
    # system = TabArchSys(s3dis.xyz_4096, [1024, 256, 64, 16], 32,  [0.1, 0.2, 0.4, 0.8])
    # system = TabArchSys(s3dis.xyz_512, [256, 64, 16], 16,  [0.2, 0.4, 0.8])
    # system = TabArchSys(s3dis.xyz_256, [128, 32, 8, 2], 32,  [0.8, 1.6, 3.2, 6.4])
    # system = TabArchSys(shapenet.xyz_512, [128, 32, 32], 16, [0.4, 0.8, 0.8])

    import time
    start = time.time()
    system.SysUnitTest()
    stop = time.time()
    print('Simulation time: ', (stop-start)/60, 'min')
    
    # DramUnitTest()
    