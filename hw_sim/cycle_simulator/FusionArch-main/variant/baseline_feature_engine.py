import math
import numpy as np
import torch
import copy
import pdb
import dramAddrParse as mem_parser
import test_data as tdat

# class Dram:

#     class DramEntry:
#         def __init__(self, request):
#             self.request = request
#             self.timer = 50

#     def __init__(self):
#         self.pending_list = []
    
#     def cycle(self, requests):
#         # append all requests to the pending list
#         if len(requests) > 0:
#             for req in requests:
#                 self.pending_list.append(self.DramEntry(req))
            
#         return_list = []
#         for i in range(len(self.pending_list)):
#             self.pending_list[i].timer -= 1
#             if self.pending_list[i].timer == 0:
#                 return_list.append(self.pending_list[i].request)
            
#         # pop the request from the pending list when its timer is 0
#         self.pending_list = list(filter(lambda req:req.timer>0, self.pending_list))

#         return return_list

#     def debug_show_dram(self):
#         output = {}
#         for req in self.pending_list:
#             output[req.request] = req.timer
#         print(output)

class FeatureComputingEngine:
    def __init__(self, slice_num, core_num, hashtab_size, xyz):
        self.nslice = slice_num
        self.entity = [FeatureComputingSlice(i, core_num, hashtab_size, xyz) for i in range(self.nslice)]
        self.dram_response = []

        # self.in_channels_list = [3, 128+3]
        # self.out_channels_list = [128, 256] 
        self.in_channels_list = [3, 128+3, 256+3]
        self.out_channels_list = [128, 256, 1024] 
        
        # self.mlp_conv_macs = [[0, 0, 0], [0, 0, 0]]
        # self.mlp_bn_macs = [[0, 0, 0], [0, 0, 0]]
        self.mlp_conv_macs  = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.mlp_bn_macs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.sa_layerid = 0     # the current sa layer id
        self.sa_layernum = len(self.out_channels_list)    # the max number of sa layers in pointnet2
        
        # # only for debug
        # self.centroids = tdat.centroids_128# contains all centroids of point cloud (include each cluster's centroids (indices))
        # self.neighbors = tdat.neighbors_128_2 # contains each cluster's each centroid's neighbors (indices)
        # self.clusters = tdat.centroids_128  # contains which centroids belong to which cluster
        # self.sa_layerid = 1
        # self.sa_layernum = 1 
        # self.is_centroids_ready = True
        # self.is_neighbors_ready = True

        self.centroids = None # contains all centroids of point cloud (include each cluster's centroids (indices))
        self.neighbors = None # contains each cluster's each centroid's neighbors (indices)
        self.clusters = None  # contains which centroids belong to which cluster
        self.is_centroids_ready = False
        self.is_neighbors_ready = False
        
        
        self.pending_read_centroids = []
        self.pending_read_neighbors = []
        # self.pending_read_clusters = []
        
        # self.is_slice_allocated = torch.zeros(self.nslice)
        self.count_allocated_slices = 0
        self.count_complete_slices = 0
        self.internal_timer = 0
        # self.dram = Dram()
        # self.requests = []

    # clusters: clustering result
    def input_centroids(self, centroids):
        requests_list = []
        if self.centroids is None and self.count_allocated_slices == 0:
            addrs = mem_parser.gen_centroids_addr_inbatch(centroids)
            self.pending_read_centroids = [int(x, 16) for x in addrs]
            for addr in addrs:
                requests_list.append([addr, 0, 3])
            self.centroids = centroids
            self.clusters = centroids
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
    
    # def input_clusters(self, clusters):
    #     requests_list = []
    #     if self.clusters is None and self.count_allocated_slices == 0:
    #         self.clusters = [sublist for sublist in clusters if sublist]
    #         for i in range(len(self.clusters)):
    #             addrs = mem_parser.gen_cluster_addr(len(self.clusters[i]))
    #             self.pending_read_clusters.extend([int(x, 16) for x in addrs])
    #             for addr in addrs:
    #                 requests_list.append([addr, 0, 3])
    #     return requests_list
    
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
                    print('[FC - input centroids] sa_layerid: {}'.format(self.sa_layerid))
                    break
            elif res in self.pending_read_neighbors:
                self.pending_read_neighbors.remove(res)
                # print('remove a request from pending_read_neighbors: ', res, ', len(pending_read_neighbors): ', len(self.pending_read_neighbors))
                self.dram_response.remove([res, False])
                if len(self.pending_read_neighbors) == 0:
                    self.is_neighbors_ready = True
                    print('[FC - input neighbors] sa_layerid: {}'.format(self.sa_layerid))
                    break
            # elif res in self.pending_read_clusters:
            #     self.pending_read_clusters.remove(res)
            #     # print('remove a request from pending_read_clusters: ', res, ', len(pending_read_clusters): ', len(self.pending_read_clusters))
            #     self.dram_response.remove((res, False))
            #     if len(self.pending_read_clusters) == 0:
            #         self.is_clusters_ready = True
            #         print('[FC - input clusters] sa_layerid: {}'.format(self.sa_layerid))
            #         break
        
        for i in range(self.nslice):
            # if self.admission[i] == False:
            state, requests = self.entity[i].cycle(self.dram_response)
            requests_list.extend(requests)
            
            if state == True:
                self.count_complete_slices += 1
            
            # if self.neighbors is not None and (self.is_centroids_ready and self.is_neighbors_ready) is True:
            if self.is_centroids_ready and self.is_neighbors_ready:
                if self.entity[i].empty() and len(self.clusters) > 0:
                    print('[Slice - Allocation] allocate new tasks to slice', i)
                    self.count_allocated_slices += 1
                    centroids = self.clusters[:max(min(len(self.clusters), len(self.centroids)//self.nslice), 1)]
                    self.clusters = self.clusters[len(centroids):]
                    neighbors = torch.index_select(torch.tensor(self.neighbors), 0, torch.tensor([i for i, x in enumerate(self.centroids) if x in centroids])).tolist()
                    self.entity[i].input(centroids, neighbors, self.in_channels_list[self.sa_layerid-1], self.out_channels_list[self.sa_layerid-1], self.mlp_conv_macs[self.sa_layerid-1], self.mlp_bn_macs[self.sa_layerid-1])

        if self.count_complete_slices == self.count_allocated_slices and self.count_allocated_slices > 0:
            # the current SA layers' tasks has been allocated completely
            if self.sa_layerid < self.sa_layernum:
                # self.sa_layerid += 1
                self.centroids = None # TODO: set value for the next SA layer
                self.neighbors = None
                self.clusters = None
                self.is_centroids_ready = False
                self.is_neighbors_ready = False
                # self.is_clusters_ready = False
                self.count_allocated_slices = 0
                is_idle = -1    # complete 1 SA layer, waiting for the new clustering result
            else:   
                is_idle = 1     # all SA layers' tasks has been finished


        # # dram simulation
        # if len(self.requests) > 0:
        #     req = self.requests.pop(0)
        #     ret = self.dram.cycle(req)
            
        #     if len(ret) > 0:
        #         self.dram_response.append([int(ret[0], 16), False if ret[1] == 0 else True])
        
        self.internal_timer += 1
        return is_idle, requests_list
    
    # # Unit Test
    # def UnitTest(self):
    #     while True:
    #         fc_state = self.cycle()
    #         if fc_state == 1:
    #             print('SA {} finished at {}'.format(self.sa_layerid, self.internal_timer))
    #             print('[System Report] Feature Computing finished, time consuming: ', self.internal_timer)
    #             fc_state = True
    #             break
    # Unit Test
    def UnitTest(self):
        total_layernum = 3
        layerid = 0
        npoints = [512, 128, 128]

        import test_data as tdat
        centroids = tdat.centroids
        neighbors = tdat.neighbors
        clusters  = tdat.clusters

        self.input_centroids(centroids[:npoints[layerid]])
        self.input_neighbors(neighbors[layerid])
        # self.input_clusters(clusters[layerid])

        while True:
            state = self.cycle()

            if state == -1 and layerid < total_layernum:
                print('SA {} finished at {}'.format(layerid, self.internal_timer))
                layerid += 1
                self.input_centroids(centroids[:npoints[layerid]])
                self.input_neighbors(neighbors[layerid])
                # self.input_clusters(clusters[layerid])
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
        self.max_rehash_times = 5
        self.lhtentry_len = 10 + self.corenum + 2    # neighbor_id, core_masks, valid, prev hash function
        if math.log2(self.pointnum) != 10:  # for debug
            print('WARNING: The length of neighbor id field in LHT entry does not match the point number.')

        self.lhtentry_num = int(hashtab_size/self.lhtentry_len)
        self.vcbentry_num = 0

        # architecure entities
        self.lht = np.array([lhtEntry(self.corenum) for _ in range(self.lhtentry_num)])    # local hash table
        self.vcb = np.array([lhtEntry(self.corenum) for _ in range(self.vcbentry_num)])    # victim buffer
        self.cores = [Core(i, self.pcd) for i in range(self.corenum)]    # cores

        # data variables
        self.centroids = []     # centroids indices
        self.neighbors = []     # neighbors indices

        # intermediate queue for inter-components communication
        self.core2lht = []    # from core to local hash table
        
        # gather requests
        # self.requests_list = []
        self.stat_accessnum = 0   # statistic of i/o traffic
        self.accessnum_without_opt = 0
        
        # self.feature_dims = []  # dimension of each layers' feature
        # self.in_channels = [3, 128+3, 256+3]
        # self.out_channels = [128, 256, 1024]    # for debug, should be set in allocation 
        self.in_channels = 0
        self.out_channels = 0
        # self.sa_layerid = -1        # indicate which layer of MLPs is currently running
        # self.cores_state = torch.zeros(self.corenum)    # indicate which cores are idle

        self.internal_clock = 0 # for debug

        self.count_centroid_ldst = 0
        self.pending_read_features = []
        self.mlp_conv_macs = []
        self.mlp_bn_macs = []

    def input(self, centroids, neighbors, in_channels, out_channels, mlp_conv_macs, mlp_bn_macs):
        self.centroids = centroids
        self.neighbors = neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_conv_macs = mlp_conv_macs
        self.mlp_bn_macs = mlp_bn_macs
        self.stat_accessnum = 0 
        # self.sa_layerid = 0    # SA layer index
        self.accessnum_without_opt = torch.tensor(neighbors).numel()
        self.count_centroid_ldst = 0
    
    def sizeof_table(self, which_tab):
        tabsize = 0
        if which_tab == 'lht':  # local hash table
            for item in self.lht:
                if item.valid is True:
                    tabsize += 1
        elif which_tab == 'vic': # victim buffer
            for item in self.vcb:
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
    
    def hash1(self, value):
        return int(value % self.lhtentry_num)

    def hash2(self, value):
        return int((value // self.lhtentry_num) % self.lhtentry_num)
    
    def cuckoohash_search(self, value, mode=0):
        hash1_idx = self.hash1(value)
        hash2_idx = self.hash2(value)
        
        # check if the access can be coallesced firstly
        if self.lht[hash1_idx].neighbor_id == value:
            return 1, hash1_idx    # memory coalescing in hash1_idx
        elif self.lht[hash2_idx].neighbor_id == value:
            return 1, hash2_idx    # memory coalescing in hash2_idx
        else:                      # memory coalescing in victim buffer
            for i in range(self.vcb.size):
                if self.vcb[i].neighbor_id == value:
                    return 0, i
        
        # then check if there is empty slot in either h1 or h2
        if self.lht[hash1_idx].valid is False:
            self.lht[hash1_idx].hash_function = 1
            return 1, hash1_idx    # hash1_idx is empty 
        elif self.lht[hash2_idx].valid is False:
            self.lht[hash1_idx].hash_function = 2
            return 1, hash2_idx    # hash2_idx is empty
       
        if mode == 0:   # mode == 0, means the fuction is called to insert a new entry 
            return -1, hash1_idx
        else:   # mode == 1, means the function is called in the rehash loop to handle collision
            return -1, (hash2_idx if self.lht[hash2_idx].hash_function == 1 else hash1_idx)
    
    def cycle(self, dram_response):

        requests_list = [] # reset the requests list
        is_complete = False

        read_response = [x[0] for x in dram_response if x[1] == False and x[0] is not None]
        for res in read_response:
            # set the cooresponding core mask to 0 for each related hashtable entry
            if self.in_channels == 0:
                break
            res = mem_parser.parse_addr2featureidx(res, self.in_channels)
            
            # print('[Read Response] {}'.format(res))
            for i in range(len(self.pending_read_features)):
                if self.pending_read_features[i][0] == res:
                    if self.pending_read_features[i][1] == 0:
                        is_core_completed = self.cores[self.pending_read_features[i][2]].handle_dram_response(res)
                        if is_core_completed is True:
                            print('[Core Report] All requests from Core {} has been served at cycle {}.'.format(self.pending_read_features[i][2], self.internal_clock))
                            # print('receive neighbor {}\'s features needed by core {}'.format(res, self.pending_read_features[i][2]))
                        self.pending_read_features.remove([res, 0, self.pending_read_features[i][2]])
                    else:
                        self.pending_read_features[i][1] -= 1
                    break

        for c in range(self.corenum):
            if self.cores[c].empty() and len(self.neighbors) > 0 and len(self.centroids) > 0:
                print('[Core Allocation] Allocate centriod {} and its neighbors to slice {}\'s core {} at {}.'.format(self.centroids[0], self.sliceid, c, self.internal_clock))
                self.count_centroid_ldst += 1
                self.cores[c].allocate(self.centroids.pop(0), self.neighbors.pop(0), self.mlp_conv_macs, self.mlp_bn_macs)
                
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
                addrs = mem_parser.gen_feature_addr(request, self.in_channels)
                self.pending_read_features.append([request, len(addrs), c])
                for addr in addrs:
                    requests_list.append([addr, 0, 3])
        
        self.internal_clock += 1
        return is_complete, requests_list


class lhtEntry:

    def __init__(self, cores):
        self.neighbor_id = -1
        self.core_masks = '0'*cores
        self.valid = False
        self.hash_function = 1
    
    def release(self):
        self.neighbor_id = -1
        self.core_masks = '0'*len(self.core_masks)
        self.valid = False
        self.hash_function = 1
    
    def is_collision(self, request):
        label = False
        if self.valid is False: # empty entry
            label = True
        else:
            if self.neighbor_id == request: # memory coalescing
                label = True
        return not label

    # insert or coalese
    def cycle(self, request, coreid):
        self.neighbor_id = request
        self.core_masks = self.core_masks[:coreid] + '1' + self.core_masks[coreid+1:]
        self.valid = True
    
    # replace (handle kick out)
    def replace(self, new_neighbor_id, new_core_masks, new_hash_function):
        current_entry = copy.deepcopy(self)
        self.neighbor_id = new_neighbor_id
        self.core_masks = new_core_masks
        self.hash_function = new_hash_function
        if self.valid is False:
            self.valid = True
            return None
        else:
            return current_entry
    
    def replace_with_coreid(self, new_neighbor_id, new_core_id):
        current_entry = copy.deepcopy(self)
        self.neighbor_id = new_neighbor_id
        self.core_masks = self.core_masks[:new_core_id] + '1' + self.core_masks[new_core_id+1:]
        self.hash_function = 1
        if self.valid is False:
            self.valid = True
            return None
        else:
            return current_entry
    
    def toggle_hash_function(self):
        self.hash_function = 1 if self.hash_function == 2 else 2 

class Core:
    
    def __init__(self, coreid, xyz):
        self.coreid = coreid
        self.pcd = xyz  # global memory data

        # label variables
        # self.can_issue = False
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

        # intermediate queues between two components
        self.mularr2normact = []     # the queue between multiplier array and normalize and activation

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
        # self.can_issue = False          # reset the issue flag
        
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
                # pdb.set_trace()
                self.local_fpsum[(self.local_fpsum[:,0]==data_back).nonzero().squeeze(), 1] = True
            # remove the returned data (index) from the receiver list
            # self.__response_queue = np.setdiff1d(self.__response_queue, np.array(data_back))
            # all neighbors of the centroid have returned and the core can be released
            # if self.__response_queue.size == 0 and self.__request_queue.size == 0:
        return self.response_counter == len(self.neighbor_id) # check if all requested neighbors have returned

    def cycle(self):
        is_idle = False
        ret = None
        request = None
        # if len(self.mularr2normact) > 0:
        #     print('In NormAct of Compute Engine')
        #     self.local_fpsum[i][3] += 1
        #     # self.normact_latency_counter += 1
        #     if self.local_fpsum[i][3] >= self.normact_latency:
        #     # if self.normact_latency_counter >= self.normact_latency:
        #         # self.normact_latency_counter = 0
        #         # local_feat = self.mularr2normact.pop()
        #         local_feat = self.local_fpsum[i][0]
                
        #         if self.finished_neighbors >= len(self.neighbor_id):
        #             # neighbor (idx) 'local_feat' complete the feature extraction
        #             self.io_traffic += self.fpsum_dims[self.layerid] # write features back to the global memory
        #             self.layerid += 1
        #             if self.layerid == len(self.fpsum_dims):
        #                 return True, None # all neighbors of the current centroid are finished, the core can be released!
        #             # return True, None # all neighbors of the current centroid are finished, the core can be released!
        
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
                            # self.request_queue = None     # release the core
                            # self.response_queue = None    # release the core
                            # self.layerid = 0
                            self.reset_for_mlp_layer_switch()
                            # return True, None   # all neighbors of the current centroid are finished, the core can be released!
                break
        
        # Lightweight Multiplier Array
        # traverse local fpsums buffer, and select a ready neighbor to extract features
        for i in range(self.local_fpsum.shape[0]):
            if self.local_fpsum[i][1].item() == 1 and self.local_fpsum[i][2].item() > 0:
                self.local_fpsum[i][2] -= self.nmuls
                break  # only one neighbor can be selected in each cycle


        # Aggregation Engine
        # print('ptr: {}, len(self.neighbor_id): {}'.format(self.ae_neighbors_ptr, len(self.neighbor_id)))
        if self.ae_neighbors_ptr < len(self.neighbor_id):
            # neighbor = self.pcd[int(self.neighbor_id[self.ae_neighbors_ptr])]
            self.io_traffic += 3    # load neighbor coordinate (3-dimension) from the global memory
            # diff = neighbor - self.centroid

            # hold all 'diff's on chip, each entry consists of (neighbor_id, is ready, conv2d_timer, normact_timer)
            self.local_fpsum = torch.tensor([[self.neighbor_id[self.ae_neighbors_ptr], False, self.mlp_convs_macs[0], self.mlp_normact_macs[0]]]) if self.local_fpsum.shape[0] == 0 else torch.cat((self.local_fpsum, torch.tensor([self.neighbor_id[self.ae_neighbors_ptr], False, self.mlp_convs_macs[0], self.mlp_normact_macs[0]]).view(1,4)), dim=0)
            assert self.local_fpsum.shape[0] <= len(self.neighbor_id), 'local_fpsum buffer overflow!'
            
            if self.neighbor_id[self.ae_neighbors_ptr] in self.response_queue:
                self.local_fpsum[-1][1] = True  # both neighbor id and its feature are ready

            # if self.local_fpsum.shape[0] < self.max_fpsum[self.layerid]:
            #     # hold on chip if local fpsum buffer is not full
            #     if self.local_fpsum.shape[0] == 0:
            #         self.local_fpsum = torch.tensor([[self.neighbor_id[self.ae_neighbors_ptr], False, 0, 0]])  # neighbor_id, is ready, layerid, timer(after ready)
            #     else:
            #         self.local_fpsum = torch.cat((self.local_fpsum, torch.tensor([self.neighbor_id[self.ae_neighbors_ptr], False, 0, 0]).view(1,2)), dim=0)  # stored pair: (neighbor idx, is ready)
            #     if self.neighbor_id[self.ae_neighbors_ptr] in self.__response_queue:
            #         self.local_fpsum[-1][1] = True  # both neighbor id and its feature are ready
            # else:
            #     # else write back to the global memory and then fetch it when needed
            #     self.io_traffic += 3    # write back to the global memory    
            
            self.ae_neighbors_ptr += 1
            # if self.ae_neighbors_ptr >= len(self.neighbor_id):
            #     self.ae_neighbors_ptr = 0          
        
        # IO interface (TODO: integrate return signals)
        # Both requests and responses are about features of neighbor points!!!
        if self.request_queue is not None and self.request_queue.size > 0:      
            # print('dbg-5: self.request_queue = ', self.request_queue, ', self.requestQ_ptr = ', self.requestQ_ptr, ', self.can_issue = ', self.can_issue)
            # print('In IO interface, self.__request_queue.size = ', self.__request_queue.size, ', self.__requestQ_ptr = ', self.__requestQ_ptr)   
            # whether the request send at previous cycle can be issued
            # if self.can_issue is True:
            #     # remove the sent request (at previous cycle) from the queue
            #     self.request_queue = np.delete(self.request_queue, self.requestQ_ptr)
            # else:
            #     self.requestQ_ptr += 1
            #     # self.can_issue = True # for debug, should be deleted later
            
            # if self.requestQ_ptr >= self.request_queue.size:
            #     self.requestQ_ptr = 0
                # request queue is empty which means all requests have been issued
                # if self.__request_queue.size == 0:
                #     return -1   
            if self.request_queue.size > 0:
                # print('TEST： self.requestQ_ptr = ', self.requestQ_ptr, ', self.request_queue = ', self.request_queue)
                # neigh_access = self.request_queue.pop(0)   # get the next neighbor and request it
                
                # print('dbg-core: len(self.request_queue) = ', len(self.request_queue), ', self.requestQ_ptr = ', self.requestQ_ptr, ', neigh_access = ', neigh_access)
                # pdb.set_trace()
                # ret = neigh_access  # send a new request
                request = self.request_queue[0]  # send a read request for neighbor coordinates
                self.request_queue = np.delete(self.request_queue, 0)
                # print('Send a new request: ', ret)
            # neigh_access = self.__request_queue[self.__requestQ_ptr] 
            # if np.isin(neigh_access, self.__response_queue) is False:
            #     self.__response_queue = np.append(self.__response_queue, neigh_access)  # re-add neigh_access to __response_queue
                # return False, neigh_access # send a new reqeust
        # no request to send, which means:
        #  - case1: All requests have been issued, but some have not yet been returned;
        #  - case2: exception
        # return -1
        # return False, ret
        return is_idle, ret, request

# class Dram:

#     class DramEntry:
#         def __init__(self, request):
#             self.request = request
#             self.timer = 200

#     def __init__(self):
#         self.pending_list = []
    
#     def cycle(self, requests):
#         # append all requests to the pending list
#         if len(requests) > 0:
#             for req in requests:
#                 self.pending_list.append(self.DramEntry(req))
            
#         return_list = []
#         for i in range(len(self.pending_list)):
#             self.pending_list[i].timer -= 1
#             if self.pending_list[i].timer == 0:
#                 return_list.append(self.pending_list[i].request)
            
#         # pop the request from the pending list when its timer is 0
#         self.pending_list = list(filter(lambda req:req.timer>0, self.pending_list))

#         return return_list

#     def debug_show_dram(self):
#         output = {}
#         for req in self.pending_list:
#             output[req.request] = req.timer
#         print(output)


# Unit Test
# def CoreUnitTest(centroid, neighbors, xyz):
#     # configuration parameters
#     ncores = 1
#     nmuls = 6

#     core = Core(0, nmuls, xyz)
#     core.allocate(centroid, neighbors, [128,128,256])

#     while True:
#         state, ret = core.cycle()
#         if state is False:
#             if ret is not None:
#                 core.handle_dram_response(ret)
#         else:
#             print('all neighbors are finished')
#             break

if __name__ == '__main__':
    import data
    pcd = data.xyz_1024
    hashtab_size = 128
    feature_computing_engine = FeatureComputingEngine(8, 16, hashtab_size*(10+16+2), pcd)
    feature_computing_engine.UnitTest()