import torch
import numpy as np
import pdb
from collections import OrderedDict
from print_log import OUTPUT
from print_log import DeviceType
import merge_sorter as sorter
from dram import DataSrc

class FeatureDecomposer:
    def __init__(self, seg_num):
        self.unit_parallelism = 2
        self.seg_num = seg_num
        self.__features = [[] for _ in range(seg_num)]
        self.__pending_queue = []
        
        self.__sorted_features = [[] for _ in range(seg_num)]
        self.__sorter = sorter.MergeSorter()
        self.__sorting_seg_idx = 0
        
        self.__median = np.zeros(seg_num, dtype=np.float32)
        self.__residual = [[] for _ in range(seg_num)]
        self.__features2 = [[] for _ in range(seg_num)]
        self.__seg_size = np.zeros(seg_num, dtype=np.int32)
        self.minus_parallelism = 32
        
        self.state = True
        self.finished_count = 0
        self.clock = 0
        
    
    def reset(self):
        self.__sorting_seg_idx = 0
        self.finished_count = 0
        self.clock = 0
        
        
    def __sizeof_segmented_feats(self, inp_list, dim=0):
        index_list = []
        valid_seg_num = 0
        if dim == 0:
            index_list = np.nonzero(inp_list)[0]
            valid_seg_num = len(index_list)
        elif dim == 1:
            for index, sublist in enumerate(inp_list):
                sublist_length = 0
                if isinstance(sublist, list):
                    sublist_length = len(sublist)
                elif isinstance(sublist, (np.ndarray, torch.Tensor)):
                    sublist_length = sublist.shape[0]
                if sublist_length > 0:
                    index_list.append(index)
                    valid_seg_num += 1
        return valid_seg_num, index_list
        
        
    def launch(self, features, seg_idx):
        features = features.reshape(-1) # (C*nsample*seg_size)
        self.__features2[seg_idx] = features
        self.__seg_size[seg_idx] = len(features)
        self.state = False
        if self.__sizeof_segmented_feats(self.__features, 1)[0] < self.unit_parallelism:
            self.__features[seg_idx] = features
            OUTPUT('[launch (in features queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
        else:
            self.__pending_queue.append((features, seg_idx))
            OUTPUT('[launch (in pending queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
        self.clock += 1

    def get_state(self):
        return self.state
    
    def cycle(self):
        valid_median_num, median_index_list = self.__sizeof_segmented_feats(self.__median, 0)
        if valid_median_num > 0:
            for i in range(valid_median_num):
                seg_idx = median_index_list[i]
                median = self.__median[seg_idx]
                feature_slice = self.__features2[seg_idx][:self.minus_parallelism]
                self.__features2[seg_idx] = self.__features2[seg_idx][min(len(self.__features2[seg_idx]), self.minus_parallelism):]
                self.__residual[seg_idx].extend(feature_slice - median)
                
                if len(self.__residual[seg_idx]) == self.__seg_size[seg_idx]:
                    mid = self.__median[seg_idx]
                    self.__median[seg_idx] = 0
                    residual = self.__residual[seg_idx]
                    self.__residual[seg_idx] = []
                    self.finished_count += 1
                    if self.finished_count == self.seg_num:
                        self.state = True   # Err: other seg_idx may not be finished
                        OUTPUT('Feature decomposition finished, seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
                    return (seg_idx, mid, residual)
        
        valid_sorted_num, sorted_index_list = self.__sizeof_segmented_feats(self.__sorted_features, 1)
        if valid_sorted_num > 0:
            for i in range(valid_sorted_num):
                seg_idx = sorted_index_list[i]
                median_index = int(len(self.__sorted_features[seg_idx]) / 2)
                self.__median[seg_idx] = self.__sorted_features[seg_idx][median_index]
                self.__sorted_features[seg_idx] = []
                OUTPUT('[median calculation finished] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
        
        valid_feats_num, feats_index_list = self.__sizeof_segmented_feats(self.__features, 1)
        if valid_feats_num > 0 and self.__sorter.get_state() is True:
            for i in range(valid_feats_num):
                seg_idx = feats_index_list[i]
                feature_slice = self.__features[seg_idx]
                self.__sorter.launch(feature_slice, dim=0)
                self.__sorting_seg_idx = seg_idx
                self.__features[seg_idx] = []
                
                if self.__sizeof_segmented_feats(self.__features, 1)[0] < self.unit_parallelism:
                    if len(self.__pending_queue) > 0:
                        features, seg_idx = self.__pending_queue.pop(0)
                        self.__features[seg_idx] = features
                        OUTPUT('[pending queue -> feature queue] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
                    else:
                        break
                break
        
        ret = self.__sorter.cycle()
        
        if ret[0] is not None:
            self.__sorted_features[self.__sorting_seg_idx] = ret[0]
            OUTPUT('[sorting finished] seg_idx: {}, clock: {}'.format(self.__sorting_seg_idx, self.clock), devtype=DeviceType.feature_decomposer)
        
        if self.state is False: 
            self.clock += 1


def FeatureDecomposer_UnitTest():
    seg_num = 3
    feat_decomposer = FeatureDecomposer(seg_num)
    
    import data.data as data
    xyz = data.xyz_1024
    xyz = xyz.permute(1, 0).reshape(3, 1024//32, 32).numpy()
    
    feat_decomposer.launch(xyz[:,:,0], 0)
    counter = 0
    
    while True:
        if feat_decomposer.clock == 3:
            feat_decomposer.launch(xyz[:,:,1], 1)
        elif feat_decomposer.clock == 6:
            feat_decomposer.launch(xyz[:,:,2], 2)
        
        ret = feat_decomposer.cycle()
        if ret is not None:
            counter += 1
        if counter == seg_num:
            break

# TODO: functional simulation remains to be implemented
class FeatureAggregation:
    def __init__(self, seg_num, nsample):
        self.nsample = nsample
        self.seg_num = seg_num
        self.seg_size = 0
        self.in_channels = 0
        # self.seg_state = None
        
        # self.__centroid_index = None
        self.__neighbor_index = None
        
        # self.__centroid_feats = None
        # self.__neighbor_feats = None
        # self.__feats_dim = 0
        
        # memory I/O interface
        self.__load_rqst_send_per_cycle = 2
        self.__handle_response_data_per_cycle = 2
        self.__load_rqst_queue = []     # keep load requests (to dram)
        self.__load_data_queue = []     # keep load responses (from dram)
        self.dram_response = []
        self.__neighbor_count_list = []
        
        # self.__diff_queue = None
        self.__prev_ret = []
        self.__ret_count = 0

        self.state = True
        self.clock = 0
    
    
    def reset(self):
        self.clock = 0
        self.__ret_count = 0
        self.__prev_ret = []
    
    def detect_fag_state(self):
        return self.state
    
    # send read requests to dram (mode='request') or receice read response from dram (mode='response')
    def read_port(self, mode='request', data=None):
        if mode == 'request':
            nrqst = min(self.__load_rqst_send_per_cycle, len(self.__load_rqst_queue))
            tmp_rqst = self.__load_rqst_queue[:nrqst]
            rqst = [(x, self.in_channels*4, DataSrc.feature_aggregation_unit, False) for x in tmp_rqst]
            self.__load_rqst_queue = self.__load_rqst_queue[nrqst:]
            return rqst if len(rqst) > 0 else None
        elif mode == 'response':
            if data is None:
                return
            data = [x[0] for x in data if x[0] is not None]
            if len(data) == 0:
                return 
            self.__load_rqst_queue = [x for x in self.__load_rqst_queue if x not in data] # TODO
            if isinstance(data, (int, float, torch.Tensor)):
                self.__load_data_queue.append(data)
            elif isinstance(data, list):
                self.__load_data_queue.extend(data)
            else:
                print('Err: invalid data type of read response!')
                exit(1)
        else:
            print('Err: invalid mode of read port!')
            exit(1)
    
    
    def send_centroid(self, npoint, in_channels, centroid_set):
        self.npoint = npoint
        self.seg_size = npoint // self.seg_num
        # self.seg_state = np.ones(self.seg_num, dtype=np.int32) * self.seg_size
        self.__load_rqst_queue.extend(centroid_set)
        self.__load_rqst_queue = list(OrderedDict.fromkeys(self.__load_rqst_queue))    # remove duplicates
        self.in_channels = in_channels
        # if centroid_feats is not None:
        #     self.__centroid_feats = np.array(centroid_feats) if isinstance(centroid_feats, list) else centroid_feats
        #     self.__feats_dim = len(centroid_feats[0])
        #     self.__diff_queue = np.zeros((npoint, self.nsample, self.__feats_dim))
        OUTPUT('[Aggregation] centroid indices have been loaded, clock: {}'.format(self.clock), devtype=DeviceType.feature_aggregation_unit)
    
    
    def send_neighbor(self, neighbor_list):
        self.__neighbor_index = np.array(neighbor_list)
        self.__neighbor_count_list = np.ones_like(neighbor_list, dtype=np.int32)
        self.__load_rqst_queue.extend([x for sublist in neighbor_list for x in sublist])    # 2D -> 1D
        self.__load_rqst_queue = list(OrderedDict.fromkeys(self.__load_rqst_queue))    # remove duplicates
        self.state = False
        # if neighbor_feats is not None:
        #     self.__neighbor_feats = np.array(neighbor_feats) if isinstance(neighbor_feats, list) else neighbor_feats
        OUTPUT('[Aggregation] neighbor indices have been loaded, clock: {}'.format(self.clock), devtype=DeviceType.feature_aggregation_unit)
    
    def __terminal_condition(self):
        flatten_ncl = np.sum(self.__neighbor_count_list, 1)
        segmented_ncl = np.array([np.sum(flatten_ncl[i:i+self.seg_size]) for i in range(0, len(flatten_ncl), self.seg_size)])
        gathered_seg_idx = np.where(segmented_ncl == 0)[0]
        return gathered_seg_idx

    
    def cycle(self):
        read_rqst = []
        ret = None
        
        # receive read response from dram
        if len(self.dram_response) > 0:
            self.read_port('response', self.dram_response)
            self.dram_response = []
        
        # nonzero_indices = np.where(np.sum(self.__diff_queue, axis=(1,2)))[0]
        # if len(nonzero_indices) > 0:
        #     # omit concat operation
        #     OUTPUT('nonzero_indices: {}, clock: {}'.format(nonzero_indices, self.clock)) 
        #     self.__diff_queue[nonzero_indices,:,:] = np.zeros((self.nsample, self.__feats_dim))
        #     self.state = True  
        #     self.clock += 1
        #     return True, nonzero_indices
        
        if len(self.__load_data_queue) > 0:
            resp = self.__load_data_queue[:self.__handle_response_data_per_cycle]
            self.__load_data_queue = self.__load_data_queue[min(self.__handle_response_data_per_cycle, len(self.__load_data_queue)):]
            for r in resp:
                self.__neighbor_count_list[self.__neighbor_index == r] = 0
            ret = self.__terminal_condition()
            if ret.size > 0:
                # gathered_seg_indices = self.__terminal_condition()
                # if gathered_seg_indices.size > 0:
                #     print('dbg: prev_ret: {}, gathered_seg_indices: {}'.format(self.__prev_ret, gathered_seg_indices))
                if ret.size == self.seg_num:
                    self.state = True 
                    OUTPUT('[Aggregation] The aggregation of all segments has been finished, clock: {}'.format(self.clock), devtype=DeviceType.feature_aggregation_unit)
                    
                if self.__prev_ret is not None:
                    # make a diff between prev_ret and ret
                    ret = np.setdiff1d(ret, self.__prev_ret)
                # else:
                #     ret = gathered_seg_indices
                # print('ret: {}'.format(ret))
                self.__prev_ret.extend(ret)
                

                # np.add.at(self.__neighbor_count_list, all_idx, -1)
                # centroid_complete_gather = np.where(self.__neighbor_count_list == 0)[0]
                # if len(centroid_complete_gather) > 0:
                #     self.__diff_queue[centroid_complete_gather,:,:] = self.__neighbor_feats[centroid_complete_gather] - np.array(self.__centroid_feats[centroid_complete_gather][:,np.newaxis,:])

        
        if self.state is False:            
            self.clock += 1
            
        if len(self.__load_rqst_queue) > 0:
            read_rqst = self.read_port(mode='request')
        
        return ret, read_rqst if len(read_rqst) > 0 else None   


def FeatureAggregation_UnitTest():
    import data.test_data as data
    centroid = data.centroids_demo
    neighbor = data.neighbors_demo[0]
    
    feat_aggregator = FeatureAggregation(8, 2)
    
    import dram
    mem = dram.FixedLatencyDram()
    resp = []
    counter = 0
    npoint = len(centroid)
    
    feat_aggregator.send_centroid(centroid, data.centroids_feat)
    feat_aggregator.send_neighbor(neighbor, data.neighbors_feat)
    
    while True:
        state, rqst = feat_aggregator.cycle(resp)
        if state is False:
            mem.push(rqst)
        else:
            counter = len(rqst)
            if counter == npoint:
                break
        resp = mem.cycle()

import quantization_module as qt
import copy
# feature computation
# for mid, perform conv2d, bn and relu
# for residual, perform conv
class FeatureComputation:
    def __init__(self, seg_num):
        self.seg_num = seg_num
        # unit parallelism is 1, process 1 seg per time
        self.pending_queue = []
        
        self.quant_prec_generator = qt.QuantPrecGenerator(self.seg_num)
        self.quantization = qt.QuantizationUnit(self.seg_num)
        self.dequantization = qt.DequantizationUnit(self.seg_num)
        
        self.__mid_queue = [[] for _ in range(self.seg_num)]
        self.__residual_queue = [[] for _ in range(self.seg_num)]
        self.__residual_queue_for_quant_prec_gen = [[] for _ in range(self.seg_num)]
        
        self.quant_bits = [0 for _ in range(self.seg_num)]
        self.quant_residual = [[] for _ in range(self.seg_num)]
        
        self.conved_mid_queue = [[] for _ in range(self.seg_num)]
        self.conved_residual_queue = [[] for _ in range(self.seg_num)]
        
        self.relued_mid_queue = [[] for _ in range(self.seg_num)]
        self.relued_residual_queue = [[] for _ in range(self.seg_num)]
        self.complete_seg_count = 0
        
        self.state = True
        self.clock = 0
    
    
    def reset(self):
        self.complete_seg_count = 0
        self.clock = 0
    
    
    def launch(self, seg_idx, mid, residual):
        if self.state is True:
            self.__mid_queue[seg_idx] = mid
            self.__residual_queue[seg_idx] = residual
            self.__residual_queue_for_quant_prec_gen[seg_idx] = copy.deepcopy(residual)
            self.state = False
            OUTPUT('[FCU-launch] launch in working queue, seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        else:
            self.pending_queue.append((seg_idx, mid, residual))
            OUTPUT('[FCU-launch] launch in pending queue, seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
            
    
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
        
    
    def cycle(self):
        mid_complete_seg_num, mid_complete_seg_indices = self.sizeof_queue(self.relued_mid_queue, dim=1)
        if mid_complete_seg_num > 0:
            residual_complete_seg_num, residual_complete_seg_indices = self.sizeof_queue(self.relued_residual_queue, dim=1)
            intersection = [x for x in mid_complete_seg_indices if x in residual_complete_seg_indices]
            if len(intersection) > 0:
                for inter in intersection:
                    complete_seg_idx = inter
                    self.relued_mid_queue[complete_seg_idx] = []
                    self.relued_residual_queue[complete_seg_idx] = []
                
                OUTPUT('[FCU] seg feature computation finished, seg_idx: {}, clock: {}'.format(complete_seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
                
                self.state = True
                if len(self.pending_queue) > 0:
                    seg_idx, mid, residual = self.pending_queue.pop(0)
                    self.launch(seg_idx, mid, residual)
                             
                self.complete_seg_count += len(intersection)
                if self.complete_seg_count == self.seg_num:
                    OUTPUT('[FCU] feature computation finished, clock: {}'.format(self.clock), devtype=DeviceType.feature_computation_unit)
                    return (True, intersection)
                
                return (False, intersection)
            
        
        # perform bn&relu for mid
        conved_mid_num, conved_mid_indices = self.sizeof_queue(self.conved_mid_queue, dim=1)
        if conved_mid_num > 0:
            seg_idx = conved_mid_indices.pop(0)
            conved_mid = self.conved_mid_queue[seg_idx]
            self.conved_mid_queue[seg_idx] = []
            self.relued_mid_queue[seg_idx] = conved_mid # temp policy TODO
            OUTPUT('[FCU] mid bn&relu start... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        
        # perform bn&relu for residual
        conved_residual_num, conved_residual_indices = self.sizeof_queue(self.conved_residual_queue, dim=1)
        if conved_residual_num > 0:
            seg_idx = conved_residual_indices.pop(0)
            conved_residual = self.conved_residual_queue[seg_idx]
            self.conved_residual_queue[seg_idx] = []
            self.relued_residual_queue[seg_idx] = conved_residual # temp policy TODO
            OUTPUT('[FCU] residual bn&relu start... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        
        # perform conv2d for mid
        mid_num, mid_indices = self.sizeof_queue(self.__mid_queue, dim=1)
        if mid_num > 0:
            seg_idx = mid_indices.pop(0)
            mid = self.__mid_queue[seg_idx]
            self.__mid_queue[seg_idx] = []
            self.conved_mid_queue[seg_idx] = mid # temp policy TODO
            OUTPUT('[FCU] mid conv start... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        
        # perform conv2d for residual
        quant_r_num, quant_r_indices = self.sizeof_queue(self.quant_residual, dim=1)
        if quant_r_num > 0:
            seg_idx = quant_r_indices.pop(0)
            quant_residual = self.quant_residual[seg_idx]
            self.quant_residual[seg_idx] = []
            self.conved_residual_queue[seg_idx] = quant_residual # temp policy TODO
            OUTPUT('[FCU] residual conv start... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
            # TODO: self.clock += res_conv_time
            # self.feature_compute_unit.send_residual(quant_residual)
        
        # quant residual
        quanted_residual = self.quantization.cycle()
        if quanted_residual is not None:
            seg_idx, q_r = quanted_residual
            self.quant_residual[seg_idx] = q_r
            OUTPUT('[FCU] quant residual finished... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        
        # fetch out the quant prec and send residual to quant
        qb_num, qb_indices = self.sizeof_queue(self.quant_bits, dim=0)
        if qb_num > 0:
            seg_idx = qb_indices.tolist().pop(0)
            quant_bits = self.quant_bits[seg_idx]
            self.quant_bits[seg_idx] = 0
            self.quantization.launch(self.__residual_queue[seg_idx], seg_idx, quant_bits)
            OUTPUT('[FCU] quant residual start... seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)
        
        # perform quantization precision generation
        qpg_ret = self.quant_prec_generator.cycle()
        if qpg_ret is not None: # complete the quant prec gen.
            seg_idx, quant_bits = qpg_ret
            self.quant_bits[seg_idx] = quant_bits
            OUTPUT('[QuantPrecGen] seg_idx: {}, quant_bits: {}, clock: {}'.format(seg_idx, quant_bits, self.clock), devtype=DeviceType.feature_computation_unit)
        
        res_seg_num, res_indices = self.sizeof_queue(self.__residual_queue_for_quant_prec_gen, 1)
        if res_seg_num > 0:
            for i in range(res_seg_num):
                seg_idx = res_indices[i]
                self.quant_prec_generator.launch(self.__residual_queue_for_quant_prec_gen[seg_idx], seg_idx, self.seg_num)
                self.__residual_queue_for_quant_prec_gen[seg_idx] = []
                OUTPUT('[QuantPrecGen] launch to quantization precision generator, seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.feature_computation_unit)

        if self.state is False:
            self.clock += 1

if __name__ == '__main__':
    # FeatureDecomposer_UnitTest()
    FeatureAggregation_UnitTest()