import torch
import numpy as np
import pdb
import data.data as data
import time
from configparser import ConfigParser
import ast

# import sampling_engine as fps
import neighbor_search_engine as knn
import morton_code_generator as mcg
import merge_sorter as sorter
import feature_computation as fcu
import dram

from print_log import OUTPUT
from print_log import DeviceType
from dram import DataSrc
from dram import IOType


def progress_bar(iter, upper_bound, timer):
    scale = upper_bound
    a = "*" * iter
    b = "." * (scale - iter)
    c = (iter / scale) * 100
    print("\r{:^3.0f}%[{}->{}] {:.2f} cycle".format(c,a,b,timer),end = "")
    # time.sleep(0.01)

class System:
    def __init__(self, xyz, config):
        # point cloud data info
        self.xyz = xyz
        self.npoint = ast.literal_eval(config.get('Network', 'npoint'))   # centroid number
        nsample = int(config.get('Network', 'nsample'))
        self.radius = ast.literal_eval(config.get('Network', 'radius'))
        self.seg_num = int(config.get('Network', 'seg_num'))
        self.seg_size = [x // self.seg_num for x in self.npoint if x is not None]
        self.mlp_in_channels = ast.literal_eval(config.get('Network', 'mlp_in_channels'))
        
        # self.fps_unit = fps.SamplingEngine(1, xyz, self.npoint[0])
        self.knn_unit = knn.NeighborSearchEngine(nsample, 1, xyz, self.radius[0], self.npoint[0])
        self.morton_code_generator = mcg.MortonCodeGenerator()
        self.morton_code_sorter = mcg.MortonCodeSorter()
        self.feature_aggregation_unit = fcu.FeatureAggregation(self.seg_num, nsample)
        self.feature_decomposer = fcu.FeatureDecomposer(self.seg_num)
        self.feature_compute_unit = fcu.FeatureComputation(self.seg_num)    
        self.onchip_buffer = dram.OnChipBuffer()
        self.dram = dram.FixedLatencyDram()
        
        # layer info
        self.current_layer = 1
        self.nlayer = int(config.get('Network', 'layer_num'))     # total layer number 
        
        # memory I/O 
        data_bus_width = int(config.get('Hardware', 'data_bus_width'))
        self.load_rqst_parallelism = data_bus_width // 32
        self.load_resp_parallelism = data_bus_width // 32
        self.load_rqst_queue = []
        # self.load_resp_queue = []
        
        # component state
        # self.fps_state = False
        self.knn_state = False
        self.mcg_state = False
        self.mcs_state = False
        self.fag_state = False
        self.fdp_state = False
        self.fcu_state = False
        self.sys_state = False
        
        # FPS unit
        self.centroids = np.arange(0, len(self.xyz))
        
        # KNN unit
        self.centroid_ptr = 0
        self.neighbors = [[] for _ in range(self.nlayer)]

        # morton code generator
        self.morton_code = [[] for _ in range(self.nlayer)]
        
        # morton code sorter
        self.sorted_idx = [[] for _ in range(self.nlayer)]
        
        # feature aggregation unit
        self.each_segment_size = np.zeros((self.nlayer, self.seg_num))
        
        # feature decomposition
        self.gathered_seg_queue = [[] for _ in range(self.nlayer)]
        self.mid_queue = [[[] for i in range(self.seg_num)] for j in range(self.nlayer)]
        self.residual_queue = [[[] for i in range(self.seg_num)] for j in range(self.nlayer)]

        self.sysclock = 560596
        
    
    # def reset_fps(self, layerid):
    #     layerid += 1
    #     if layerid < self.nlayer:
    #         self.fps_unit = False
    #         OUTPUT('[FPS] Reset FPS unit, go ahead to layer {}'.format(layerid), devtype=DeviceType.system)
    #     else:
    #         OUTPUT('[FPS] All layers\' FPS have been finished', devtype=DeviceType.system)

    
    def layer_switch(self):
        self.current_layer += 1
        if self.current_layer < self.nlayer:
            OUTPUT('\n[SYS] Layer {} finished, forward to layer {}, sysclock: {}\n'.format(self.current_layer-1, self.current_layer, self.sysclock), devtype=DeviceType.system)
            # turn off all states
            # self.fps_state = False
            self.knn_state = False
            self.mcg_state = False
            self.mcs_state = False
            self.fag_state = False
            self.fdp_state = False
            self.fcu_state = False
            self.sys_state = False
            
            prelayer_centroids = self.centroids
            # self.xyz = self.xyz[prelayer_centroids]
            self.centroid_ptr = 0
            if self.npoint[self.current_layer] is not None:
            #     # self.fps_unit.reset(self.xyz, self.npoint[self.current_layer])
                self.knn_unit.update_pcd(self.xyz, prelayer_centroids, self.radius[self.current_layer], self.npoint[self.current_layer])
            self.morton_code_generator.reset()
            self.feature_decomposer.reset()
            self.feature_compute_unit.reset()
            self.feature_aggregation_unit.reset()

        else:
            OUTPUT('[SYS] All layers have been processed, sysclock: {}'.format(self.sysclock), devtype=DeviceType.system)
            self.sys_state = True
            self.sysclock += 1
        
    
    def sizeof_queue(self, queue, dim=0):
        index_list = []
        valid_seg_num = 0
        if dim == 0:
            index_list = np.nonzero(queue)[0]
            valid_seg_num = len(index_list)
        elif dim == 1:
            for index, sublist in enumerate(queue):
                sublist_length = 0
                if isinstance(sublist, list):
                    sublist_length = len(sublist)
                elif isinstance(sublist, (np.ndarray, torch.Tensor)):
                    sublist_length = sublist.shape[0]
                if sublist_length > 0:
                    index_list.append(index)
                    valid_seg_num += 1
        return valid_seg_num, index_list
    
    
    def read_port(self, mode=IOType.response, data=None):
        if mode == IOType.response:
            data = [x for x in data if x is not None]
            for resp in data:
                self.onchip_buffer.response(resp)
                data_field, data_src, is_write = resp
                
                # if data_src == DataSrc.fps_unit:
                #     self.fps_unit.dram_response.append((data_field, is_write))
                if data_src == DataSrc.knn_unit:
                    self.knn_unit.dram_response.append((data_field, is_write))
                elif data_src == DataSrc.morton_code_generator:
                    self.morton_code_generator.dram_response.append((data_field, is_write))
                elif data_src == DataSrc.feature_aggregation_unit:
                    self.feature_aggregation_unit.dram_response.append((data_field, is_write))
            
                
    def cycle(self):
        # receive response from dram
        self.read_port(IOType.response, self.dram.cycle())
        
        # feature computation
        seg_num, indices = self.sizeof_queue(self.mid_queue[self.current_layer], dim=1)
        if seg_num > 0:
            seg_idx = indices.pop(0)
            
            mid = self.mid_queue[self.current_layer][seg_idx]
            residual = self.residual_queue[self.current_layer][seg_idx]
            
            self.mid_queue[self.current_layer][seg_idx] = []
            self.residual_queue[self.current_layer][seg_idx] = []
            self.feature_compute_unit.launch(seg_idx, mid, residual)
        
        ret = self.feature_compute_unit.cycle()
        if ret is not None:
            is_complete, complete_seg_indices = ret
            self.fcu_state = is_complete
            if is_complete is True:
                OUTPUT('[FCU] The feature computation of all segments are finished, duration: {}, sysclock: {}'.format(self.feature_compute_unit.clock, self.sysclock), devtype=DeviceType.system)
                self.layer_switch()
                if self.sys_state is True:
                    return self.sys_state

        # feature decomposer
        # if self.fag_state is True and self.fdp_state is False:
        if len(self.gathered_seg_queue[self.current_layer]) > 0 or self.feature_decomposer.state is False:
            ret = self.feature_decomposer.cycle()
            if ret is not None:
                seg_idx, mid, residual = ret
                self.mid_queue[self.current_layer][seg_idx].append(mid)
                self.residual_queue[self.current_layer][seg_idx].append(residual)
                if self.feature_decomposer.get_state() is True:
                    self.fdp_state = True
                    OUTPUT('[FDP] All segments have been decomposed, layerid: {}, duration: {}, sysclock: {}'.format(self.current_layer, self.feature_decomposer.clock, self.sysclock), devtype=DeviceType.system)
                    
            if len(self.gathered_seg_queue[self.current_layer]) > 0:
                seg_idx = self.gathered_seg_queue[self.current_layer].pop(0)
                seg_neighbors = np.array(self.xyz)[self.neighbors[self.current_layer][seg_idx*self.seg_size[self.current_layer]:self.seg_size[self.current_layer]*(seg_idx+1)]]
                self.feature_decomposer.launch(seg_neighbors, seg_idx)   
            
        # feature aggregation unit 
        if self.knn_state is True and self.mcs_state is True and self.fag_state is False:
            if self.feature_aggregation_unit.detect_fag_state() is True:
                # self.centroids[self.current_layer] = np.array(self.centroids[self.current_layer])[self.sorted_idx[self.current_layer]].tolist()
                # self.neighbors[self.current_layer] = np.array(self.neighbors[self.current_layer])[self.sorted_idx[self.current_layer]].tolist()
                # print('dbg: npoint: {}, cetroid: {}, neighbor: {}'.format(self.npoint[self.current_layer], self.centroids, self.neighbors[self.current_layer]))
                self.feature_aggregation_unit.send_centroid(self.npoint[self.current_layer], self.mlp_in_channels[self.current_layer], self.centroids)
                self.feature_aggregation_unit.send_neighbor(self.neighbors[self.current_layer])
            ret, rqst = self.feature_aggregation_unit.cycle()
            if rqst is not None:
                self.load_rqst_queue.extend(rqst)
            if ret is not None and len(ret) > 0:
                self.gathered_seg_queue[self.current_layer].extend(ret)
                if self.feature_aggregation_unit.detect_fag_state() is True:
                    self.fag_state = True
                    OUTPUT('[FAG] layerid: {}, duration: {}, sysclock: {}'.format(self.current_layer, self.feature_aggregation_unit.clock, self.sysclock), devtype=DeviceType.system)
        
        # morton code sorter
        if self.mcg_state is True and self.mcs_state is False:
            if self.morton_code_sorter.detect_mcs_state() is True:
                self.morton_code_sorter.launch(self.morton_code[self.current_layer])
            sorted_index = self.morton_code_sorter.cycle()
            if sorted_index is not None:
                self.mcs_state = True
                self.sorted_idx[self.current_layer] = sorted_index
                OUTPUT('[MCS] layerid: {}, duration: {}, sysclock: {}'.format(self.current_layer, self.morton_code_sorter.clock, self.sysclock), devtype=DeviceType.system)
                OUTPUT('[MCS] sorted idx: {}'.format(self.sorted_idx[self.current_layer].tolist()))
        
        # morton code generator
        if self.mcg_state is False:
            if self.morton_code_generator.detect_mcg_state() is True:
                self.morton_code_generator.launch(self.xyz)
            rqst = self.morton_code_generator.cycle()
            if rqst is not None and len(rqst) > 0:
                self.load_rqst_queue.extend(rqst)
            if self.morton_code_generator.detect_mcg_state() is True:
                self.mcg_state = True
                self.morton_code[self.current_layer] = self.morton_code_generator.get_morton_code()
                OUTPUT('[MCG] layerid: {}, morton_code_num: {}, duration: {}, sysclock: {}'.format(self.current_layer, len(self.morton_code[self.current_layer]), self.morton_code_generator.clock, self.sysclock), devtype=DeviceType.system)
        
        # neighbor search (knn/ball query, default: ball query)
        if self.knn_state is False:
            centroid = None
            if self.centroid_ptr < len(self.centroids):
                centroid = self.centroids[self.centroid_ptr]
                self.centroid_ptr += 1
            is_complete, neighbor, rqst = self.knn_unit.cycle(centroid)
            progress_bar(len(self.neighbors[self.current_layer]), self.npoint[self.current_layer], self.knn_unit.clock) # progress bar
            if len(neighbor) > 0:
                self.neighbors[self.current_layer].extend(neighbor)
            if len(rqst) > 0:
                self.load_rqst_queue.extend(rqst)
            if is_complete is True:
                self.knn_state = True
                OUTPUT('[KNN] layerid: {}, duration: {}, sysclock: {}'.format(self.current_layer, self.knn_unit.clock, self.sysclock), devtype=DeviceType.system)
                OUTPUT('[KNN] neighbors: {}'.format(self.neighbors[self.current_layer]), devtype=DeviceType.system)
        
        # # farthest sampling point
        # if self.npoint[self.current_layer] is not None and len(self.centroids[self.current_layer]) < self.npoint[self.current_layer]:
        #     is_complete, centroid, rqst = self.fps_unit.cycle()
        #     progress_bar(len(self.centroids[self.current_layer]), self.npoint[self.current_layer], self.fps_unit.clock) # progress bar
        #     if centroid is not None:
        #         self.centroids[self.current_layer].append(centroid)
        #     if len(rqst) > 0:
        #         self.load_rqst_queue.extend(rqst)
        #     if is_complete is True:
        #         self.fps_state = True
        #         OUTPUT('[FPS] layerid: {}, npoint: {}, duration: {}, sysclock: {}'.format(self.current_layer, len(self.centroids[self.current_layer]), self.fps_unit.clock, self.sysclock), devtype=DeviceType.system)
        #         OUTPUT('[FPS] centroids: {}'.format(self.centroids[self.current_layer]), devtype=DeviceType.system)
                
        # send request to dram
        if len(self.load_rqst_queue) > 0:
            nrqst = min(self.load_rqst_parallelism, len(self.load_rqst_queue))
            rqst = self.load_rqst_queue[:nrqst]
            self.load_rqst_queue = self.load_rqst_queue[nrqst:]
            buffer_ret = self.onchip_buffer.request(rqst)
            if len(buffer_ret) > 0:
                self.read_port(IOType.response, buffer_ret)
            rqst = [x for x in rqst if dram.isin(x, buffer_ret) is False]
            if len(rqst) > 0:
                self.dram.push(rqst)
        
        self.sysclock += 1
        

if __name__ == '__main__':
    config = ConfigParser()
    # import data.test_data as data
    # xyz = data.xyz
    # config.read('./config/demo.ini')
    
    # ---------- ModelNet40 ----------
    # import data.modelnet as data
    # idx = torch.randperm(1024)[:256]
    # xyz = data.xyz_1024[idx]
    # config.read('./config/dgcnn_cls_x16.ini')
    
    # # ---------- ShapeNet ----------
    # import data.shapenet as data
    # xyz = data.xyz_512
    # config.read('./config/dgcnn_ps_x64.ini')
    
    # # ---------- ShapeNet ----------
    import data.s3dis as data
    xyz = data.xyz_512
    config.read('./config/dgcnn_s_x16.ini')
    
    sys = System(xyz, config)
    
    while True:
        state = sys.cycle()
        if state is True:
            break