import math
import numpy as np
import torch
import copy
import pdb
import dramAddrParse as mem_parser
import cache
import test_data

class FeatureComputingEngine:
    def __init__(self, slice_num, core_num, hashtab_size, xyz):
        self.nslice = slice_num
        self.entity = [FeatureComputingSlice(i, core_num, hashtab_size, xyz) for i in range(self.nslice)]
        self.dram_response = []

        # self.in_channels_list = [3, 128+3, 256+3]
        # self.out_channels_list = [128, 256, 1024] 
        self.in_channels_list = [256+3]
        self.out_channels_list = [1024] 
        # self.in_channels_list = [9 + 3, 64 + 3, 128 + 3, 256 + 3]
        # self.out_channels_list = [64, 128, 256, 512]  
        # self.in_channels_list = [3, 16, 32, 64]
        # self.out_channels_list = [16, 32, 64, 128]  
        # self.in_channels_list = [3, 3, 3]
        
        # self.mlp_conv_macs  = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # self.mlp_bn_macs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.mlp_conv_macs  = [[0, 0, 0]]
        self.mlp_bn_macs = [[0, 0, 0]]

        self.sa_layerid = 0     # the current sa layer id
        self.sa_layernum = len(self.out_channels_list)    # the max number of sa layers in pointnet2

        # only for debug
        self.centroids = test_data.centroids
        self.neighbors = test_data.neighbors[0]
        self.clusters = test_data.clusters[0]
        self.avg_alloc = math.ceil(sum([len(x) for x in self.clusters])/self.nslice)

        # self.centroids = None # contains all centroids of point cloud (include each cluster's centroids (indices))
        # self.neighbors = None # contains each cluster's each centroid's neighbors (indices)
        # self.clusters = None  # contains which centroids belong to which cluster
        # self.is_centroids_ready = False
        # self.is_neighbors_ready = False
        # self.is_clusters_ready = False
        # self.avg_alloc = 0

        # for debug
        self.is_centroids_ready = True
        self.is_neighbors_ready = True
        self.is_clusters_ready = True
        self.sa_layerid = 1
        self.sa_layernum = 1

        self.pending_read_centroids = []
        self.pending_read_neighbors = []
        self.pending_read_clusters = []
        
        self.count_allocated_slices = 0
        self.count_complete_slices = 0
        self.internal_timer = 0
        
        self.is_clustering_balanced = False

        self.fixed_dram = Dram()
        self.requests = []

    # clusters: clustering result
    def input_centroids(self, centroids):
        requests_list = []
        if self.centroids is None and self.count_allocated_slices == 0:
            addrs = mem_parser.gen_centroids_addr_inbatch(centroids)
            self.pending_read_centroids = [int(x, 16) for x in addrs]
            for addr in addrs:
                requests_list.append([addr, 0, 3])
            self.centroids = centroids
        return requests_list

    def input_neighbors(self, neighbors):
        requests_list = []
        if self.neighbors is None and self.count_allocated_slices == 0:
            addrs = mem_parser.gen_neighbors_addr_inbatch(neighbors)
            self.pending_read_neighbors = [int(x, 16) for x in addrs]
            for addr in addrs:
                requests_list.append([addr, 0, 3])
            self.neighbors = neighbors

        return requests_list
    
    def input_clusters(self, clusters):
        requests_list = []
        if self.clusters is None and self.count_allocated_slices == 0:
            self.clusters = clusters
            self.avg_alloc = math.ceil(sum([len(x) for x in self.clusters])/self.nslice)

            for i in range(len(self.clusters)):
                addrs = mem_parser.gen_cluster_addr(len(self.clusters[i]))
                self.pending_read_clusters.extend([int(x, 16) for x in addrs])
                for addr in addrs:
                    requests_list.append([addr, 0, 3])
        return requests_list
    
    def get_clusterid_with_maxpts(self):
        delta_cluster_size = [len(x)-self.avg_alloc for x in self.clusters]
        max_size = max(delta_cluster_size)
        clusterid_with_maxpts = delta_cluster_size.index(max_size)
        min_size = min(delta_cluster_size)
        clusterid_with_minpts = delta_cluster_size.index(min_size)
        return clusterid_with_maxpts, clusterid_with_minpts
    
    def cycle(self):
        requests_list = []
        is_idle = 0
        # ret = None

        read_response = [x[0] for x in self.dram_response if x[1] == False]
        for res in read_response:
            if res in self.pending_read_centroids:
                self.pending_read_centroids.remove(res)
                # print('remove a request from pending_read_centroids: ', res, ', len(pending_read_centroids): ', len(self.pending_read_centroids))
                self.dram_response.remove([res, False])
                if len(self.pending_read_centroids) == 0:
                    self.is_centroids_ready = True
                    self.sa_layerid += 1
                    self.count_complete_slices = 0
                    print('[FC - input centroids] sa_layerid: {}, cycle: {}'.format(self.sa_layerid, self.internal_timer))
                    break
            elif res in self.pending_read_neighbors:
                self.pending_read_neighbors.remove(res)
                # print('remove a request from pending_read_neighbors: ', res, ', len(pending_read_neighbors): ', len(self.pending_read_neighbors))
                self.dram_response.remove([res, False])
                if len(self.pending_read_neighbors) == 0:
                    self.is_neighbors_ready = True
                    print('[FC - input neighbors] sa_layerid: {}, cycle: {}'.format(self.sa_layerid, self.internal_timer))
                    break
            elif res in self.pending_read_clusters:
                self.pending_read_clusters.remove(res)
                # print('remove a request from pending_read_clusters: ', res, ', len(pending_read_clusters): ', len(self.pending_read_clusters))
                self.dram_response.remove([res, False])
                if len(self.pending_read_clusters) == 0:
                    self.is_clusters_ready = True
                    print('[FC - input clusters] sa_layerid: {} cycle: {}'.format(self.sa_layerid, self.internal_timer))
                    break
        
        # allocate points in clusters to slices
        if self.clusters is not None and self.is_clusters_ready: 
            nonempty_cluster_size = [x for x in self.clusters if len(x) > self.avg_alloc]
            # if self.remain_slots.count(0) < self.nslice and [len(x) for x in self.clusters].count(0) < len(self.clusters):
            if len(nonempty_cluster_size) > 0:
                maxid, minid = self.get_clusterid_with_maxpts()
                # sliceid_with_maxslots = self.get_sliceid_with_maxslots()
                # npts_scheduled = min(self.remain_slots[sliceid_with_maxslots], len(self.clusters[clusterid_with_maxpts]))
                npts_scheduled = min(len(self.clusters[maxid])-self.avg_alloc, self.avg_alloc-len(self.clusters[minid]))
                self.clusters[minid].extend(self.clusters[maxid][-npts_scheduled:])
                self.clusters[maxid] = self.clusters[maxid][:-npts_scheduled]
                # self.sliceslots[sliceid_with_maxslots].extend(self.clusters[clusterid_with_maxpts][:npts_scheduled])
                # self.clusters[clusterid_with_maxpts] = self.clusters[clusterid_with_maxpts][npts_scheduled:]
                # self.remain_slots[sliceid_with_maxslots] = self.remain_slots[sliceid_with_maxslots] - npts_scheduled
                print('clusters: {}, time: {}'.format([len(x) for x in self.clusters], self.internal_timer))
            else:
                self.is_clustering_balanced = True
                self.clusters = [x for x in self.clusters if len(x)>0]
                # print('clusters-2: {}'.format(self.clusters))
        
        for i in range(self.nslice):
            state, requests = self.entity[i].cycle(self.dram_response)
            requests_list.extend(requests)
            if state == True:
                self.count_complete_slices += 1
            
        # if self.clusters is not None and self.neighbors is not None and (self.is_centroids_ready and self.is_clusters_ready and self.is_neighbors_ready) is True:
        #     if self.entity[i].empty() and len(self.clusters) > 0 and len(self.clusters[0]) > 0:
        #         self.count_allocated_slices += 1
                # centroids = self.clusters.pop(0)    # centroids of 1 cluster
        

            if self.neighbors is not None and (self.is_centroids_ready and self.is_neighbors_ready) and self.is_clustering_balanced:
                if self.entity[i].empty() and len(self.clusters) > 0 and len([x for x in self.clusters if len(x) > self.avg_alloc]) == 0:
                    self.count_allocated_slices += 1
                    centroids = self.clusters.pop(0)
                    # self.sliceslots[allocidx] = []
                    # self.remain_slots[allocidx] = self.nslots_per_slice
                    print('[Slice - Allocation] allocate centroids {} to Slice {} at cycle {}'.format(centroids, i, self.internal_timer))
                    neighbors = torch.index_select(torch.tensor(self.neighbors), 0, torch.tensor([i for i, x in enumerate(self.centroids) if x in centroids])).tolist()
                    self.entity[i].input(centroids, neighbors, self.in_channels_list[self.sa_layerid-1], self.out_channels_list[self.sa_layerid-1], self.mlp_conv_macs[self.sa_layerid-1], self.mlp_bn_macs[self.sa_layerid-1])

            if self.count_complete_slices == self.count_allocated_slices and self.count_allocated_slices > 0:
                # the current SA layers' tasks has been allocated completely
                # print('dbg: sa_layerid: {}, sa_layernum: {}'.format(self.sa_layerid, self.sa_layernum))
                if self.sa_layerid < self.sa_layernum:
                    self.centroids = None # TODO: set value for the next SA layer
                    self.neighbors = None
                    self.clusters = None
                    self.is_centroids_ready = False
                    self.is_neighbors_ready = False
                    self.is_clusters_ready = False
                    self.count_allocated_slices = 0
                    is_idle = -1    # complete 1 SA layer, waiting for the new clustering result
                    self.avg_alloc = 0
                    self.is_clustering_balanced = False
                else:   
                    is_idle = 1     # all SA layers' tasks has been finished
        
        # dram with fixed latency, for debug
        self.requests.extend(requests_list)
        if len(self.requests) > 0:
            req = self.requests.pop(0)
            res = self.fixed_dram.cycle([int(req[0], 16), False if req[1]==0 else True, req[2]])
            if len(res) > 0:
                self.dram_response.append(res[:2])

        self.internal_timer += 1
        return is_idle, requests_list

    # Unit Test
    def UnitTest(self):
        total_layernum = 1
        layerid = 1
        npoints = [512,128,128]

        import test_data as tdat
        centroids = tdat.centroids
        neighbors = tdat.neighbors
        clusters  = tdat.clusters

        self.input_centroids(centroids[:128])
        self.input_neighbors(neighbors[2])
        self.input_clusters(clusters[2])

        while True:
            state = self.cycle()

            if state == -1 and layerid < total_layernum:
                print('SA {} finished at {}'.format(layerid, self.internal_timer))
                layerid += 1
                self.input_centroids(centroids[:npoints[layerid]])
                self.input_neighbors(neighbors[layerid])
                self.input_clusters(clusters[layerid])
                print('Probing: len(centroids): {}, len(neighbors): {}, len(clusters): {}'.format(len(centroids), len(neighbors), [len(x) for x in clusters]))
            if state == 1:
                print('SA {} finished at {}'.format(layerid, self.internal_timer))
                print('[System Report] Feature Computing finished, time consuming: ', self.internal_timer)
                break
        

class FeatureComputingSlice:

    # ============= initialization =============
    # sliceid: slice index
    # core_num: number of cores in this slice
    # hashtab_size: size of local hash table in bits
    # xyz: point cloud data
    # ==========================================
    def __init__(self, sliceid, core_num, hashtab_size, xyz):
        self.sliceid = sliceid
        self.corenum = core_num
        self.pcd = xyz
        self.pointnum= xyz.shape[0]

        # config parameters
        # self.max_rehash_times = 5
        key_len = 10
        self.lhtentry_len = key_len + self.corenum + 1   # neighbor_id, core_masks, valid
        if math.log2(self.pointnum) > key_len:  # for debug
            print('WARNING: The length of neighbor id field in LHT entry does not match the point number.')

        self.lhtentry_num = int(hashtab_size/self.lhtentry_len)

        # architecure entities
        self.lht = np.array([lhtEntry(self.corenum) for _ in range(self.lhtentry_num)])    # local hash table
        # self.vcb = np.array([lhtEntry(self.corenum) for _ in range(self.vcbentry_num)])    # victim buffer
        self.cores = [Core(i, self.pcd) for i in range(self.corenum)]    # cores

        # data variables
        self.centroids = []     # centroids indices
        self.neighbors = []     # neighbors indices

        # intermediate queue for inter-components communication
        self.core2lht = []    # from core to local hash table
        
        # gather requests
        self.stat_accessnum = 0   # statistic of i/o traffic
        self.accessnum_without_opt = 0
        
        self.in_channels = 0
        self.out_channels = 0

        self.internal_clock = 0 # for debug

        self.count_centroid_ldst = 0
        self.pending_read_features = []
        self.mlp_conv_macs = []
        self.mlp_bn_macs = []

        self.cache = cache.cache()
        self.cache_access_cnt = 0
        self.cache_hit_cnt = 0
        self.coalesced_by_lft = 0

    def input(self, centroids, neighbors, in_channels, out_channels, mlp_conv_macs, mlp_bn_macs):
        self.centroids = centroids
        self.neighbors = neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_conv_macs = mlp_conv_macs
        self.mlp_bn_macs = mlp_bn_macs
        self.stat_accessnum = 0 
        self.accessnum_without_opt = torch.tensor(neighbors).numel()
        self.count_centroid_ldst = 0
        self.cache_access_cnt = 0
        self.cache_hit_cnt = 0
    
    def sizeof_table(self):
        tabsize = 0
        for item in self.lht:
            if item.valid is True:
                tabsize += 1
        return tabsize

    def empty(self):
        is_any_core_busy = False
        for i in range(self.corenum):
            if self.cores[i].empty() is False:
                is_any_core_busy = True
                break
        return not is_any_core_busy and len(self.core2lht)==0 and len(self.centroids)==0
    
    def lut_put(self, neighbor_id, src):
        # detect whether the neighbor can be coalesced
        empty_slot = -1
        state = -1   # -1: fail to handling the request; 0: memory coalesce; 1: create a new entry and insert into the LUT
        is_coalesced = False
        for id in range(self.lhtentry_num):
            slot = self.lht[id]
            if slot.valid is False:
                if empty_slot == -1:
                    empty_slot = id
            else:
                if slot.neighbor_id == neighbor_id:
                    # memory coalescing
                    self.lht[id].set_bit(src)
                    is_coalesced = True
                    state = 0
                    break
        
        if is_coalesced is False:
            if empty_slot >= 0:
                # create a new entry and insert it into the LUT
                self.lht[empty_slot].neighbor_id = neighbor_id
                self.lht[empty_slot].set_bit(src)
                self.lht[empty_slot].valid = True
                state = 1
            else:
                # LUT is full, cannot insert the new entry
                state = -1

        return state

    def lut_get(self, request):
        state = -1   # is find 
        for id in range(self.lhtentry_num):
            if self.lht[id].valid is True:
                if self.lht[id].neighbor_id == request:
                    state = id
                    break
        return state
    
    def cycle(self, dram_response):

        requests_list = [] # reset the requests list
        is_complete = False

        read_response = [x[0] for x in dram_response if x[1] == False and x[0] is not None]
        for res in read_response:
            # set the cooresponding core mask to 0 for each related hashtable entry
            if self.in_channels == 0:
                break
            res = mem_parser.parse_addr2featureidx(res, self.in_channels)
            for i in range(len(self.pending_read_features)):
                if self.pending_read_features[i][0] == res and self.pending_read_features[i][1] > 0:
                    self.pending_read_features[i][1] -= 1
                    if self.pending_read_features[i][1] == 0:
                        self.pending_read_features.remove([res, 0])
                        self.cache.put(res)

                        state = self.lut_get(res)
                        if state >= 0:
                            core_masks = self.lht[state].core_masks
                            self.lht[state].release()
                        
                        if core_masks != '' and core_masks.count('1') > 0:
                            for bid in range(len(core_masks)):
                                bit = core_masks[bid]
                                if bit == '1':
                                    is_core_completed = self.cores[bid].handle_dram_response(res)
                                    if is_core_completed is True:
                                        print('[Core Report] All requests from Core {} has been served at cycle {}.'.format(bid, self.internal_clock))
                    break

        # process up to 16 (i.e, self.corenum) requests from different cores in 1 cycle
        while len(self.core2lht) > 0:  
            c, request = self.core2lht.pop()
            if request >= 0:
                if self.cache.get(request) is True:
                    self.cache_access_cnt += 1
                    self.cache_hit_cnt += 1
                    # print('[Cache Hit] Core {} access {}'.format(c, request))
                    self.cores[c].can_issue = True

                    is_core_completed = self.cores[c].handle_dram_response(request)
                    if is_core_completed is True:
                        print('[Core Report] All requests from Core {} has been served at cycle {}.'.format(c, self.internal_clock))

                else:   # cache miss, then create an entry in LHT
                    slot = self.lut_put(request, c)
                    self.cores[c].can_issue = True
                    if slot == -1:
                        # the LUT is full, fail to sending the request
                        self.cores[c].can_issue = False
                    elif slot == 1:
                        self.cache_access_cnt += 1
                        addrs = mem_parser.gen_feature_addr(request, self.in_channels)
                        self.pending_read_features.append([request, len(addrs)])
                        for addr in addrs:
                            requests_list.append([addr, 0, 3])
                        self.stat_accessnum += 1        
                    else: # state == 0, memory request is coalesced
                        self.coalesced_by_lft += 1                

        for c in range(self.corenum):
            if self.cores[c].empty() and len(self.neighbors) > 0 and len(self.centroids) > 0:
                self.count_centroid_ldst += 1
                neighbors = self.neighbors.pop(0)
                first_neighbor = neighbors[0]
                neighbors = [x for x in neighbors if x != first_neighbor]
                neighbors.insert(0, first_neighbor)
                print('[Core Allocation] Allocate centriod {} and its neighbors (size: {}) to slice {}\'s core {} at cycle {}.'.format(self.centroids[0], len(neighbors), self.sliceid, c, self.internal_clock))

                self.cores[c].allocate(self.centroids.pop(0), neighbors, self.mlp_conv_macs, self.mlp_bn_macs)
                # print('left centroids: {}, left neighbors: {}'.format(len(self.centroids), len(self.neighbors)))

            core_state, ret_centroid, request = self.cores[c].cycle() # remains to implement dramsim
            if ret_centroid is not None and core_state is True:
                print('[Core Report] Slice {} Core {} is released at cycle {}'.format(self.sliceid, c, self.internal_clock))
                addrs = mem_parser.gen_feature_addr(ret_centroid, self.out_channels)
                for addr in addrs:
                    requests_list.append([addr, 1, 3])
                self.count_centroid_ldst -= 1
                if self.count_centroid_ldst == 0 and len(self.centroids) == 0:
                    # all centroids in the cluster have been processed
                    is_complete = True

            if request is not None and request >= 0:
                self.core2lht.append([c, request])
        
        if is_complete is True and self.accessnum_without_opt > 0:
            print('[Slice Report] Slice {} finished simulation! Reduced access: {}%, cache hit rate: {}%'.format(self.sliceid, (self.accessnum_without_opt-self.stat_accessnum)/self.accessnum_without_opt*100, self.cache_hit_cnt/self.cache_access_cnt*100))
            self.accessnum_without_opt = 0
        
        self.internal_clock += 1
        self.cache.cycle()
        return is_complete, requests_list

class lhtEntry:

    def __init__(self, cores):
        self.neighbor_id = -1
        self.core_masks = '0'*cores
        self.valid = False
        # self.hash_function = 1
    
    def release(self):
        self.neighbor_id = -1
        self.core_masks = '0'*len(self.core_masks)
        self.valid = False

    def set_bit(self, src):
        self.core_masks = self.core_masks[:src] + '1' + self.core_masks[src+1:]

class Core:
    
    def __init__(self, coreid, xyz):
        self.coreid = coreid
        self.pcd = xyz  # global memory data

        # label variables
        self.can_issue = False
        self.layerid = 0
        self.finished_neighbors = 0

        # local memory
        self.centroid_id = None      # the centroid index allocated to this core
        self.centroid = None         # the centroid coordinate 
        self.neighbor_id = []      # the neighbors indices
        self.request_queue = None  # neighbors indices, used to send request
        self.response_queue = None # neighbors indices, used to receive request
        # local memory pointers
        self.requestQ_ptr = -1     # pointer of the 'request_queue' 
        self.response_counter = 0  # count elements of the 'response_queue'
        self.ae_neighbors_ptr = 0    # pointer of the 'neighbor_id' in the aggregation engine
        # neighbor points' features
        # self.size_for_fpsum = 2**21  # the size of local memory specially for fpsum (i.e. neighbors' features)
        # self.fpsum_dims = []         # the dimensions of each layers' neighbor features
        # self.max_fpsum = []          # the maximum numbers of fpsums that can be held for each MLP layer
        self.local_fpsum = torch.tensor([])      # the local fpsums buffer, only record neighbor id in simulation, hold all 'diff's on chip, each entry consists of (neighbor_id, is ready, conv2d_timer, normact_timer)

        # the number of multipliers(macs) in the each core's multiplier array
        self.nmuls = 1024 

        # measure the io traffic of 1 core
        self.io_traffic = 0

        # Timing parameters
        self.normact_latency = 10
        self.normact_latency_counter = 0   # incremental timer
        # self.mularr_latency = 0       # incremental timer
        self.mlp_convs_macs = []    # TODO: 在allocate中赋值
        self.mlp_normact_macs = []
    
    # call this function when sa_layer + 1 or allocate to a new centroid
    def allocate(self, centroid, neighbors, mlp_conv_macs, mlp_bn_macs):
        self.io_traffic = 0             # reset the io traffic counter
        self.requestQ_ptr = -1        # reset the neighbor queue's pointer
        self.response_counter = 0     # reset the response counter
        self.ae_neighbors_ptr = 0       # reset the 'neighbor_id' pointer
        self.can_issue = False          # reset the issue flag
        
        self.mlp_layerid = 0                # reset the mlp layer id
        self.normact_latency_counter = 0  # reset the normalize and activation timer
        self.finished_neighbors = 0     # reset the label which records finished neighbors
        
        self.centroid_id = centroid
        self.centroid = self.pcd[self.centroid_id]  # get the centroid coordinate from the global memory
        self.io_traffic += 3    # load centroid coordinate (3-dimension) from the global memory 
        self.neighbor_id = neighbors
        self.request_queue = np.array(neighbors)
        self.response_queue = []
        # self.fpsum_dims = feature_dims      
        # self.max_fpsum = [self.size_for_fpsum // dim for dim in self.fpsum_dims]
        self.local_fpsum = torch.tensor([]) 
        self.mlp_convs_macs = mlp_conv_macs   # TODO: 在allocate中赋值
        self.mlp_normact_macs = mlp_bn_macs    # just for mlp layers of the current sa layer
    
    # call this function when mlp_layer + 1
    def reset_for_mlp_layer_switch(self):
        self.mlp_layerid = 0                # reset the mlp layer id
        self.normact_latency_counter = 0  # reset the normalize and activation timer
        self.finished_neighbors = 0     # reset the label which records finished neighbors
        self.local_fpsum = torch.tensor([]) 

        self.request_queue = None     # release the core
        self.response_queue = None    # release the core
    
    def empty(self):
        return self.response_queue is None and self.request_queue is None

    def handle_dram_response(self, data_back):
        if data_back is not None and self.request_queue is not None:
            self.response_counter += 1
            self.response_queue.append(data_back)
            self.io_traffic += self.mlp_convs_macs[0]
            if data_back in self.local_fpsum[:,0]:
                self.local_fpsum[(self.local_fpsum[:,0]==data_back).nonzero().squeeze(), 1] = True
        return self.response_counter == len(self.neighbor_id) # check if all requested neighbors have returned

    def cycle(self):
        is_idle = False  
        ret = None
        request = None

        # Computing Engine
        # Normalize and Activation unit
        for i in range(self.local_fpsum.shape[0]):
            # start normalize and activation, before the stage the following conditions must be satisfied: neighbor 'id' is ready (include its features), conv2d is finished (conv2d timer is 0) and normact timer is set (larger than 0)
            if self.local_fpsum[i][1].item() == 1 and self.local_fpsum[i][2].item() <= 0 and self.local_fpsum[i][3].item() >= 0:
                self.local_fpsum[i][3] -= 1
                if self.local_fpsum[i][3].item() <= 0:
                    self.finished_neighbors += 1
                    if self.finished_neighbors >= len(self.neighbor_id):
                        self.mlp_layerid += 1   # MLP layers in 1 SA layer
                        print('[Core {}] MLP layer + 1 = {}'.format(self.coreid, self.mlp_layerid))
                        if self.mlp_layerid < len(self.mlp_normact_macs):
                            for j in range(self.local_fpsum.shape[0]):
                                self.local_fpsum[j][2] = self.mlp_convs_macs[self.mlp_layerid]
                                self.local_fpsum[j][3] = self.mlp_normact_macs[self.mlp_layerid]
                        else:
                            is_idle = True
                            ret = self.centroid_id
                            print('[Core {}] All neighbors of the current centroid are finished, the core can be released!'.format(self.coreid))
                            self.reset_for_mlp_layer_switch()
                break
        
        # Lightweight Multiplier Array
        # traverse local fpsums buffer, and select a ready neighbor to extract features
        for i in range(self.local_fpsum.shape[0]):
            if self.local_fpsum[i][1].item() == 1 and self.local_fpsum[i][2].item() > 0:
                self.local_fpsum[i][2] -= self.nmuls
                break  # only one neighbor can be selected in each cycle


        # Aggregation Engine
        if self.ae_neighbors_ptr < len(self.neighbor_id):
            # neighbor = self.pcd[int(self.neighbor_id[self.ae_neighbors_ptr])]
            self.io_traffic += 3    # load neighbor coordinate (3-dimension) from the global memory
            # diff = neighbor - self.centroid

            # hold all 'diff's on chip, each entry consists of (neighbor_id, is ready, conv2d_timer, normact_timer)
            self.local_fpsum = torch.tensor([[self.neighbor_id[self.ae_neighbors_ptr], False, self.mlp_convs_macs[0], self.mlp_normact_macs[0]]]) if self.local_fpsum.shape[0] == 0 else torch.cat((self.local_fpsum, torch.tensor([self.neighbor_id[self.ae_neighbors_ptr], False, self.mlp_convs_macs[0], self.mlp_normact_macs[0]]).view(1,4)), dim=0)
            assert self.local_fpsum.shape[0] <= len(self.neighbor_id), 'local_fpsum buffer overflow!'

            if self.neighbor_id[self.ae_neighbors_ptr] in self.response_queue:
                    self.local_fpsum[-1][1] = True  # both neighbor id and its feature are ready  
            
            self.ae_neighbors_ptr += 1

        if self.request_queue is not None and self.request_queue.size > 0:      
            # whether the request send at previous cycle can be issued
            if self.can_issue is True:
                self.request_queue = np.delete(self.request_queue, self.requestQ_ptr)
            else:
                self.requestQ_ptr += 1
            
            if self.requestQ_ptr >= self.request_queue.size:
                self.requestQ_ptr = 0
                # request queue is empty which means all requests have been issued
            if self.request_queue.size > 0:
                neigh_access = self.request_queue[self.requestQ_ptr]    # get the next neighbor and request it
                request = neigh_access  # send a read request for neighbor coordinates

        # no request to send, which means:
        #  - case1: All requests have been issued, but some have not yet been returned;
        #  - case2: exception
        # return -1
        # return False, ret
        return is_idle, ret, request

class Dram:

    class DramEntry:
        def __init__(self, request):
            self.request = request
            self.timer = 50

    def __init__(self):
        self.pending_list = []
    
    def cycle(self, requests):
        # append all requests to the pending list
        if len(requests) > 0:
            for req in requests:
                self.pending_list.append(self.DramEntry(req))
            
        return_list = []
        for i in range(len(self.pending_list)):
            self.pending_list[i].timer -= 1
            if self.pending_list[i].timer <= 0:
                return_list.append(self.pending_list[i].request)
            
        # pop the request from the pending list when its timer is 0
        self.pending_list = list(filter(lambda req:req.timer>0, self.pending_list))

        return return_list

    def debug_show_dram(self):
        output = {}
        for req in self.pending_list:
            output[req.request] = req.timer
        print(output)

if __name__ == '__main__':
    slicenum = 8
    corenum = 16
    tablesize = 32
    
    import data
    pcd = data.xyz_1024

    feature_computing_engine = FeatureComputingEngine(slicenum, corenum, tablesize*(10+16+1), pcd)
    feature_computing_engine.UnitTest()
