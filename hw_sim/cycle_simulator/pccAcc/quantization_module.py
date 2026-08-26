import numpy as np
import torch
import pdb
from print_log import OUTPUT
from print_log import DeviceType
import copy

data_bus_width = 384

class QuantPrecGenerator:    
    
    def __init__(self, seg_num):
        self.component_parallelsim = 2  # handle 2 segments concurrently
        self.seg_num = seg_num
        self.scale = 0.09
        
        # segment queue
        # self.__lane_state = np.zeros(seg_num, dtype=bool)  # 0: idle, 1: busy
        self.__segments_pending_queue = []
        self.__segmented_feats = [[] for _ in range(seg_num)]
        self.__seg_size = np.zeros(seg_num, dtype=np.int32)

        # mean unit
        self.mean_parallelism = 32
        self.__sum_queue = np.zeros(seg_num, dtype=np.float32)
        self.__zp_div_queue = np.zeros(seg_num, dtype=np.float32)
        self.__mean_queue = np.zeros(seg_num, dtype=np.float32)
        
        # max/min unit
        self.__max_queue = np.zeros(seg_num, dtype=np.float32)  
        self.__min_queue = np.zeros(seg_num, dtype=np.float32)
        self.__interval_size = np.zeros(seg_num, dtype=np.float32)
        
        # confidence interval
        self.__left_interval_range = np.zeros(seg_num, dtype=np.float32)
        self.__right_interval_range = np.zeros(seg_num, dtype=np.float32)
        self.__features = [[] for _ in range(seg_num)]
        self.compare_parallelism = 32
        self.__confidence_counter = np.zeros(seg_num, dtype=np.int32)
        self.__confidence_prob = np.zeros(seg_num, dtype=np.float32)
        
        # judge precision
        self.__judge_prec_queue = np.zeros(seg_num, dtype=np.float32)
        self.__quant_bits = np.zeros(seg_num, dtype=np.int32)
        
        self.state = True
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
    
    
    def launch(self, features, seg_idx, seg_num=-1):
        # input features: (C, seg_size)
        features = np.array(features).reshape(-1) # (C*seg_size)
        self.seg_num = seg_num if seg_num > 0 else self.seg_num
        self.__features[seg_idx] = features
        if self.__sizeof_segmented_feats(self.__segmented_feats, 1)[0] < self.component_parallelsim:
            self.__segmented_feats[seg_idx] = copy.deepcopy(features)
            self.__seg_size[seg_idx] = features.shape[0]
            # self.__lane_state[seg_idx] = True
            self.state = False
            OUTPUT('[QuantPrecGen] launch (in segmented_feats_queue), seg_idx: {}, seg_size: {}, clock: {}'.format(seg_idx, self.__seg_size[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
        else:   # if full, put into pending queue
            self.__segments_pending_queue.append((features, seg_idx))
            OUTPUT('[QuantPrecGen] launch (in pending_queue), seg_idx: {}, seg_size: {}, clock: {}'.format(seg_idx, self.__seg_size[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
        self.clock += 1
    
    def cycle(self):
        
        valid_judge_num, judge_index_list = self.__sizeof_segmented_feats(self.__judge_prec_queue)
        if valid_judge_num > 0:
            for i in range(valid_judge_num):
                seg_idx = judge_index_list[i]
                prob = self.__judge_prec_queue[seg_idx]
                self.__quant_bits[seg_idx] = 4 if prob >= 0.99 else 8
                self.__judge_prec_queue[seg_idx] = 0
                OUTPUT('[QuantPrecGen] 1 seg finished, seg_idx: {}, prec: {}, clock: {}'.format(seg_idx, self.__quant_bits[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
                if np.all(self.__quant_bits > 0):
                    self.state = True
                    OUTPUT('[QuantPrecGen] Quantization precisions are determined: {}, clock: {}'.format(self.__quant_bits, self.clock), devtype=DeviceType.quant_prec_generator)
                    self.clock = 0
                    # return self.__quant_bits
                return (seg_idx, self.__quant_bits[seg_idx])
            
        # confidence_interval_count / seg_size --> confidence prob
        valid_prob_num, prob_index_list = self.__sizeof_segmented_feats(self.__confidence_prob)
        if valid_prob_num > 0:
            for i in range(valid_prob_num):
                seg_idx = prob_index_list[i]
                self.__confidence_prob[seg_idx] = self.__confidence_prob[seg_idx] / self.__seg_size[seg_idx]
                self.__judge_prec_queue[seg_idx] = self.__confidence_prob[seg_idx]
                self.__confidence_prob[seg_idx] = 0
                OUTPUT('[Confidence Count -> Confidence Prob] seg_idx: {}, prob: {}, clock: {}'.format(seg_idx, self.__judge_prec_queue[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
        
        # stat points fall in interval range4
        valid_range_num, range_index_list = self.__sizeof_segmented_feats(self.__left_interval_range)
        if valid_range_num > 0:
            for i in range(valid_range_num):
                seg_idx = range_index_list[i]
                feature_slice = self.__features[seg_idx][:self.compare_parallelism]
                self.__features[seg_idx] = self.__features[seg_idx][self.compare_parallelism:]
                self.__confidence_counter[seg_idx] += np.sum((feature_slice >= self.__left_interval_range[seg_idx]) & (feature_slice <= self.__right_interval_range[seg_idx]))
                if len(self.__features[seg_idx]) == 0:
                    self.__confidence_prob[seg_idx] = self.__confidence_counter[seg_idx]
                    self.__left_interval_range[seg_idx] = 0
                    self.__right_interval_range[seg_idx] = 0
                    self.__interval_size[seg_idx] = 0
                    self.__features[seg_idx] = []
                    OUTPUT('[Interval Range -> Confidence Count] seg_idx: {}, prob: {}, clock: {}'.format(seg_idx, self.__confidence_prob[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
                
        # mean +- interval_size --> confidence interval range
        valid_interval_num, interval_index_list = self.__sizeof_segmented_feats(self.__interval_size)
        if valid_interval_num > 0:
            for i in range(valid_interval_num):
                seg_idx = interval_index_list[i]
                if self.__mean_queue[seg_idx] != 0 and self.__left_interval_range[seg_idx] == 0:
                    self.__left_interval_range[seg_idx] = self.__mean_queue[seg_idx] - self.__interval_size[seg_idx]
                    self.__right_interval_range[seg_idx] = self.__mean_queue[seg_idx] + self.__interval_size[seg_idx]
                    OUTPUT('[Mean+-Interval -> Interval Range] seg_idx: {}, left: {}, right: {}, clock: {}'.format(seg_idx, self.__left_interval_range[seg_idx], self.__right_interval_range[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
        
        # mean unit: div list --> confidence interval: offset list
        valid_div_num, div_index_list = self.__sizeof_segmented_feats(self.__zp_div_queue)
        if valid_div_num > 0:
            for i in range(valid_div_num):
                seg_idx = div_index_list[i]
                self.__mean_queue[seg_idx] = self.__zp_div_queue[seg_idx] / self.__seg_size[seg_idx]
                self.__zp_div_queue[seg_idx] = 0
                OUTPUT('[Div -> Mean] seg_idx: {}, mean: {}, clock: {}'.format(seg_idx, self.__mean_queue[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)
        
        # mean unit: add list (sum) --> div list
        valid_seg_num, seg_index_list = self.__sizeof_segmented_feats(self.__segmented_feats, 1)
        if valid_seg_num > 0:
            for i in range(valid_seg_num):
                seg_idx = seg_index_list[i]
                feature_slice = self.__segmented_feats[seg_idx][:self.mean_parallelism]
                self.__sum_queue[seg_idx] += np.sum(feature_slice, 0)
                self.__max_queue[seg_idx] = np.max(feature_slice, 0)
                self.__min_queue[seg_idx] = np.min(feature_slice, 0)
                self.__segmented_feats[seg_idx] = self.__segmented_feats[seg_idx][self.mean_parallelism:]
                
                if len(self.__segmented_feats[seg_idx]) == 0:
                    self.__zp_div_queue[seg_idx] = self.__sum_queue[seg_idx]
                    self.__sum_queue[seg_idx] = 0
                    self.__interval_size[seg_idx] = (self.__max_queue[seg_idx] - self.__min_queue[seg_idx]) * self.scale
                    OUTPUT('[Add -> Div] seg_idx: {}, sum: {}, max-min: {}, clock: {}'.format(seg_idx, self.__zp_div_queue[seg_idx], self.__interval_size[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)

            # pending queue --> segment queue
            if self.__sizeof_segmented_feats(self.__segmented_feats, 1)[0] < self.component_parallelsim and len(self.__segments_pending_queue) > 0:
                features, seg_idx = self.__segments_pending_queue.pop(0)
                self.__segmented_feats[seg_idx] = features
                # self.__lane_state[seg_idx] = True
                self.__seg_size[seg_idx] = features.shape[0]
                OUTPUT('[pending queue -> segmented feats] seg_idx: {}, seg_size: {}, clock: {}'.format(seg_idx, self.__seg_size[seg_idx], self.clock), devtype=DeviceType.quant_prec_generator)

        if self.state is False:
            self.clock += 1


def QuantPrecGen_UnitTest():
    seg_num = 3
    qpg = QuantPrecGenerator(seg_num)
    
    import data.data as data
    xyz = data.xyz_1024     # (1024, 3)
    xyz = xyz.permute(1, 0).reshape(3, 1024//32, 32).numpy()  # (C, seg_size, seg_num)
    
    qpg.launch(xyz[:,:,0], 0)
    
    while True:
        quant_bits = qpg.cycle()
        if qpg.clock == 16:
            qpg.launch(xyz[:,:,1], 1)
        elif qpg.clock == 32:
            qpg.launch(xyz[:,:,2], 2)
        if quant_bits is not None:
            break


class QuantizationUnit:
    def __init__(self, seg_num):
        self.seg_num = seg_num
        self.unit_parallelism = 2
        self.__seg_size = np.zeros(seg_num, dtype=np.int32)
        
        self.__segmented_feats = [[] for _ in range(seg_num)]
        self.__segments_pending_queue = []
        self.__quant_bits = np.zeros(seg_num, dtype=np.int32)
        
        # compute scale
        self.__delta_r = np.zeros(seg_num, dtype=np.float32)
        self.__min_queue = np.zeros(seg_num, dtype=np.float32)
        self.__delta_q = np.zeros(seg_num, dtype=np.float32)
        self.__constant_delta_q = {4: 15, 8: 255, 16: 65535}
        self.__scale = np.zeros(seg_num, dtype=np.float32)
        
        # compute zero point
        self.__zero_point = np.zeros(seg_num, dtype=np.float32)
        self.__zp_div_queue = np.zeros(seg_num, dtype=np.float32)
        
        # quantization
        self.quantization_parallelism = 32
        self.__features = [[] for _ in range(seg_num)]
        self.__quant_div_queue = [[] for _ in range(seg_num)]
        self.__quant_round_queue = [[] for _ in range(seg_num)]
        self.__quant_add_queue = [[] for _ in range(seg_num)]
        self.__quantized_queue = [[] for _ in range(seg_num)]
        
        self.state = True
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
    
    
    def __terminal_condition(self):
        if self.__sizeof_segmented_feats(self.__quantized_queue, 1)[0] == self.seg_num:
            for seg_idx in range(self.seg_num):
                sublist = self.__quantized_queue[seg_idx]
                if len(sublist) < self.__seg_size[seg_idx]:
                    return False
            return True
        return False
        
        
    def launch(self, features, seg_idx, quant_bits):
        # input features: (C*seg_size,)
        features = np.array(features).reshape(-1) # (C*seg_size)
        self.__features[seg_idx] = features
        self.__seg_size[seg_idx] = features.shape[0]
        self.state = False
        
        if self.__sizeof_segmented_feats(self.__segmented_feats, 1)[0] < self.unit_parallelism:
            self.__segmented_feats[seg_idx] = copy.deepcopy(features)
            self.__quant_bits[seg_idx] = quant_bits
            OUTPUT('[launch (in segmented_feats_queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
        else:
            self.__segments_pending_queue.append((features, seg_idx, quant_bits))
            OUTPUT('[launch (in pending_queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
        self.clock += 1
        
    
    def cycle(self):
        # perform clip of quantized feature
        valid_add_num, add_index_list = self.__sizeof_segmented_feats(self.__quant_add_queue, 1)
        if valid_add_num > 0:
            for i in range(valid_add_num):
                seg_idx = add_index_list[i]
                feature_slice = self.__quant_add_queue[seg_idx][:self.quantization_parallelism]
                self.__quant_add_queue[seg_idx] = self.__quant_add_queue[seg_idx][self.quantization_parallelism:]
                self.__quantized_queue[seg_idx].extend(np.clip(feature_slice, 0, 2**self.__quant_bits[seg_idx]-1))
                OUTPUT('[quant clip] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
            
            # quantization finished
            if self.__terminal_condition() is True:
                self.state = True
                OUTPUT('Quantization finished, quantized_feats.shape: {}, clock: {}'.format([len(x) for x in self.__quantized_queue], self.clock), devtype=DeviceType.quant_unit)
                self.clock = 0
            return (seg_idx, self.__quantized_queue[seg_idx])
        
        # compute add (+zero point) of quantization
        valid_round_num, round_index_list = self.__sizeof_segmented_feats(self.__quant_round_queue, 1)
        if valid_round_num > 0:
            for i in range(valid_round_num):
                seg_idx = round_index_list[i]
                if self.__zero_point[seg_idx] != 0:
                    feature_slice = self.__quant_round_queue[seg_idx][:self.quantization_parallelism]
                    self.__quant_round_queue[seg_idx] = self.__quant_round_queue[seg_idx][self.quantization_parallelism:]
                    self.__quant_add_queue[seg_idx].extend(feature_slice + self.__zero_point[seg_idx])
                    OUTPUT('[quant add] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
        
        # compute round of quantization
        valid_quant_div_num, quant_div_index_list = self.__sizeof_segmented_feats(self.__quant_div_queue, 1)
        if valid_quant_div_num > 0:
            for i in range(valid_quant_div_num):
                seg_idx = quant_div_index_list[i]
                feature_slice = self.__quant_div_queue[seg_idx][:self.quantization_parallelism]
                self.__quant_div_queue[seg_idx] = self.__quant_div_queue[seg_idx][self.quantization_parallelism:]
                self.__quant_round_queue[seg_idx].extend(np.round(feature_slice))     
                OUTPUT('[quant round] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)         
        
        # compute round of zero point
        valid_zp_div_num, div_zp_index_list = self.__sizeof_segmented_feats(self.__zp_div_queue)
        if valid_zp_div_num > 0:
            for i in range(valid_zp_div_num):
                seg_idx = div_zp_index_list[i]
                self.__zero_point[seg_idx] = np.round(self.__zp_div_queue[seg_idx])
                self.__zp_div_queue[seg_idx] = 0
                OUTPUT('[zero point] seg_idx: {} zero_point: {}, clock: {}'.format(seg_idx, self.__zero_point[seg_idx], self.clock), devtype=DeviceType.quant_unit)
        
        # compute div of zero point & quantization div
        valid_scale_num, scale_index_list = self.__sizeof_segmented_feats(self.__scale)
        if valid_scale_num > 0:
            for i in range(valid_scale_num):
                seg_idx = scale_index_list[i]
                
                if self.__zero_point[seg_idx] == 0:
                    self.__zp_div_queue[seg_idx] = - self.__min_queue[seg_idx] / self.__scale[seg_idx]
                
                feature_slice = self.__features[seg_idx][:self.quantization_parallelism]
                self.__features[seg_idx] = self.__features[seg_idx][self.quantization_parallelism:]
                self.__quant_div_queue[seg_idx].extend(feature_slice / self.__scale[seg_idx])
                
                if len(self.__features[seg_idx]) == 0:
                    self.__scale[seg_idx] = 0
                    OUTPUT('[quant div] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
        
        # compute scale
        valid_seg_num, seg_index_list = self.__sizeof_segmented_feats(self.__segmented_feats, 1)
        if valid_seg_num > 0:
            for i in range(valid_seg_num):
                seg_idx = seg_index_list[i]
                self.__min_queue[seg_idx] = np.min(self.__segmented_feats[seg_idx])
                self.__delta_r[seg_idx] = np.max(self.__segmented_feats[seg_idx]) - self.__min_queue[seg_idx]
                self.__delta_q[seg_idx] = self.__constant_delta_q[self.__quant_bits[seg_idx]]
                self.__scale[seg_idx] = self.__delta_r[seg_idx] / self.__delta_q[seg_idx]
                self.__segmented_feats[seg_idx] = []
                OUTPUT('[scale] seg_idx: {}, scale: {}, clock: {}'.format(seg_idx, self.__scale[seg_idx], self.clock), devtype=DeviceType.quant_unit)
            
            # pending queue --> segmented features queue
            if self.__sizeof_segmented_feats(self.__segmented_feats, 1)[0] < self.unit_parallelism and len(self.__segments_pending_queue) > 0:
                features, seg_idx, quant_bits = self.__segments_pending_queue.pop(0)
                self.__segmented_feats[seg_idx] = features
                self.__quant_bits[seg_idx] = quant_bits
                OUTPUT('[pending queue -> segmented feats] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.quant_unit)
        
        if self.state is False:
            self.clock += 1


def Quantization_UnitTest():
    seg_num = 3
    quant = QuantizationUnit(seg_num)
    
    import data.data as data
    xyz = data.xyz_1024     # (1024, 3)
    xyz = xyz.permute(1, 0).reshape(3*1024//32, 32).numpy()
    
    quant.launch(xyz[:,0], 0, 8)
    
    while True:
        if quant.clock == 3:
            quant.launch(xyz[:,1], 1, 4)
        elif quant.clock == 5:
            quant.launch(xyz[:,2], 2, 8)
        quantized_feature = quant.cycle()
        if quantized_feature is not None:
            break
        

class DequantizationUnit:
    def __init__(self, seg_num):
        self.seg_num = seg_num
        self.unit_parallelism = 2
        
        self.__conved_feats = [[] for _ in range(seg_num)]
        self.__pending_queue = []
        self.__weights = [[] for _ in range(seg_num)]
        self.__scale = np.zeros(seg_num, dtype=np.float32)
        self.__zero_point = np.zeros(seg_num, dtype=np.float32)
        self.__kern_num = np.zeros(seg_num, dtype=np.int32)
        self.__feats_size = [[] for _ in range(seg_num)]
        
        # compute Wzp
        self.__Wzp_queue = [[] for _ in range(seg_num)]
        self.sum_parallelism = 32
        self.__weight_sum_queue = [[] for _ in range(seg_num)]
        self.__zp_mul_queue = [[] for _ in range(seg_num)]
        
        # dequantization
        self.dequantization_parallelism = 32
        self.__minus_zp_queue = [[] for _ in range(seg_num)]
        self.__dequant_queue = [[] for _ in range(seg_num)]
        
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
        
        
    def send_params(self, seg_idx, weights, scale, zero_point):
        self.__weights[seg_idx] = weights
        self.__kern_num[seg_idx] = weights.shape[0]
        self.__weight_sum_queue[seg_idx] = np.zeros(self.__kern_num[seg_idx], dtype=np.float32)
        self.__scale[seg_idx] = scale
        self.__zero_point[seg_idx] = zero_point
        OUTPUT('[send params] seg_idx: {}, scale: {}, zp: {}, clock: {}'.format(seg_idx, self.__scale[seg_idx], self.__zero_point[seg_idx], self.clock), devtype=DeviceType.dequant_unit)
        
    
    def launch(self, features, seg_idx):
        self.__feats_size[seg_idx] = features.shape
        if self.__sizeof_segmented_feats(self.__conved_feats, 1)[0] < self.unit_parallelism:
            self.__conved_feats[seg_idx] = features
            OUTPUT('[launch (in conved_feats_queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)
        else:
            self.__pending_queue.append((features, seg_idx))
            OUTPUT('[launch (in pending_queue)] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)
        self.clock += 1
        
    
    def __terminal_condition(self):
        if self.__sizeof_segmented_feats(self.__dequant_queue, 1)[0] == self.seg_num:
            return True
        return False

    def cycle(self):
        
        valid_minus_num, minus_index_list = self.__sizeof_segmented_feats(self.__minus_zp_queue, 1)
        if valid_minus_num > 0:
            for i in range(valid_minus_num):
                seg_idx = minus_index_list[i]
                self.__dequant_queue[seg_idx].extend(np.array(self.__minus_zp_queue[seg_idx]) * self.__scale[seg_idx].reshape(-1, 1))
                if len(self.__dequant_queue[seg_idx]) == self.__feats_size[seg_idx][0]:
                    self.__minus_zp_queue[seg_idx] = []
                    if self.__terminal_condition() is True:
                        OUTPUT('Dequantization finished, dequantized_feats.shape: {}, clock: {}'.format([len(x) for x in self.__dequant_queue], self.clock), devtype=DeviceType.dequant_unit)
                        self.clock = 0
                        return self.__dequant_queue
        
        # (feats - Wzp) in dequantization
        valid_wzp_num, wzp_index_list = self.__sizeof_segmented_feats(self.__Wzp_queue, 1)
        valid_feats_num, feats_index_list = self.__sizeof_segmented_feats(self.__conved_feats, 1)
        if valid_wzp_num > 0 and valid_feats_num > 0:
            for i in range(valid_feats_num):
                seg_idx = feats_index_list[i]
                if seg_idx in wzp_index_list:
                    feature_slice = self.__conved_feats[seg_idx][:,:self.dequantization_parallelism]
                    self.__conved_feats[seg_idx] = self.__conved_feats[seg_idx][:,self.dequantization_parallelism:]
                    self.__minus_zp_queue[seg_idx].extend(feature_slice - self.__Wzp_queue[seg_idx].reshape(-1, 1))                    
                    if self.__conved_feats[seg_idx].shape[1] == 0:
                        self.__Wzp_queue[seg_idx] = []
                        self.__conved_feats[seg_idx] = []
                        if len(self.__pending_queue) > 0:
                            features, seg_idx = self.__pending_queue.pop(0)
                            self.__conved_feats[seg_idx] = features
                            OUTPUT('[pending queue -> conved feats] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)
                        OUTPUT('[feats - Wzp] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)
            
        # mul zp in Wzp computation
        valid_zp_mul_num, zp_mul_index_list = self.__sizeof_segmented_feats(self.__zp_mul_queue, 1)
        if valid_zp_mul_num > 0:
            for i in range(valid_zp_mul_num):
                seg_idx = zp_mul_index_list[i]
                self.__Wzp_queue[seg_idx] = self.__zp_mul_queue[seg_idx] * self.__zero_point[seg_idx]
                self.__zp_mul_queue[seg_idx] = []
                OUTPUT('[mul zp] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)

        # sum in Wzp computation
        valid_params_num, params_index_list = self.__sizeof_segmented_feats(self.__weights, 1)
        if valid_params_num > 0:
            for i in range(valid_params_num):
                seg_idx = params_index_list[i]
                weights = self.__weights[seg_idx][:,:self.sum_parallelism]
                self.__weights[seg_idx] = self.__weights[seg_idx][:,min(len(self.__weights[seg_idx]), self.sum_parallelism):]
                self.__weight_sum_queue[seg_idx] += np.sum(weights, 1).reshape(-1)
                if self.__weights[seg_idx].shape[1] == 0:
                    self.__zp_mul_queue[seg_idx] = self.__weight_sum_queue[seg_idx]
                    self.__weights[seg_idx] = []
                    OUTPUT('[weight sum] seg_idx: {}, clock: {}'.format(seg_idx, self.clock), devtype=DeviceType.dequant_unit)
        self.clock += 1


def Dequantization_UnitTest():
    seg_num = 3
    dequant = DequantizationUnit(seg_num)
    
    import data.data as data
    xyz = data.xyz_1024     # (1024, 3)
    xyz = xyz.permute(1, 0).reshape(3, 1024//32, 32)
    
    from torch import nn
    conv = nn.Conv2d(3, 64, kernel_size=1, bias=False)
    weights = conv.state_dict()['weight'].numpy()
    conved_feats = conv(xyz)
    conved_feats = conved_feats.reshape(64, 32, 32).detach().numpy()

    dequant.send_params(0, weights, 0.038, 24)
    dequant.launch(conved_feats[:,0], 0)
    
    while True:
        if dequant.clock == 3:
            dequant.send_params(1, weights, 0.038, 24)
            dequant.launch(conved_feats[:,1], 1)
        elif dequant.clock == 5:
            dequant.send_params(2, weights, 0.038, 24)
            dequant.launch(conved_feats[:,2], 2)
        quantized_feature = dequant.cycle()
        if quantized_feature is not None:
            break

if __name__ == '__main__':
    # QuantPrecGen_UnitTest()
    # Quantization_UnitTest()
    Dequantization_UnitTest()