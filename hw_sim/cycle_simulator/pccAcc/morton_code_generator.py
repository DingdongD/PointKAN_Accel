import torch
import numpy as np
import pdb
from dram import DataSrc
from print_log import OUTPUT
from print_log import DeviceType

# data_bus_width = 384

class MortonCodeGenerator:
    def __init__(self, resolution=0.0001, bits=10, xyz=None):
        self.xyz = xyz
        self.point_num = xyz.shape[0] if xyz is not None else 0
        self.resolution = resolution
        self.bits = bits
        
        # memory I/O interface
        self.__load_rqst_send_per_cycle = 2
        self.__load_rqst_queue = xyz.cpu().tolist() if xyz is not None else []    # keep load requests (to dram)
        self.__load_data_queue = []     # keep load responses (from dram)
        self.dram_response = []
        
        # MIN unit
        self.min_parallelism = 2    # for x, y, z
        self.__min_val = np.ones(3) * float('inf')
        self.__min_cnt = 0
        self.__min_state = False    # state is Ture, finished (have found the min of x/y/z), else not finished
        
        # id generator
        self.idgen_parallelism = 2  # 2 triple, each contains (x, y, z)
        self.__idgen_add_queue = xyz.cpu().tolist() if xyz is not None else []
        self.__idgen_div_queue = []
        
        # x/y/z bit lane
        self.__x_bit_lane = []
        self.__y_bit_lane = []
        self.__z_bit_lane = []
        self.__bit_ptr = 0
        self.__pts_ptr = 0
        
        # morton code buffer
        self.__mc_buffer = np.zeros((self.point_num, self.bits), dtype=np.uint8) if xyz is not None else None
        self.mc_decval = np.zeros(self.point_num, dtype=np.uint32) if xyz is not None else None
        self.shifter_parallelism = 2 # 2x set of shifters
        
        # morton code generator internal variables
        self.mcg_state = True if xyz is None else False # idle: True, busy: False
        self.clock = 0
    
    
    def launch(self, input):
        self.xyz = input
        self.point_num = len(input)
        self.__load_rqst_queue = input
        self.__idgen_add_queue = input
        self.__mc_buffer = np.zeros((self.point_num, self.bits), dtype=np.uint8)
        self.mc_decval = np.zeros(self.point_num, dtype=np.uint32)
        self.mcg_state = False
    
    
    def reset(self):
        self.__bit_ptr = 0
        self.__pts_ptr = 0
        self.__load_data_queue = []     
        self.dram_response = []
        self.__min_val = np.ones(3) * float('inf')
        self.__min_cnt = 0
        self.__min_state = False
        self.__idgen_div_queue = []
        self.__x_bit_lane = []
        self.__y_bit_lane = []
        self.__z_bit_lane = []
        self.clock = 0
        
    
    # send read requests to dram (mode='request') or receice read response from dram (mode='response')
    def read_port(self, mode='request', data=None):
        if mode == 'request':
            nrqst = min(self.__load_rqst_send_per_cycle, len(self.__load_rqst_queue))
            tmp_rqst = self.__load_rqst_queue[:nrqst]
            rqst = [(x, 12, DataSrc.morton_code_generator, False) for x in tmp_rqst]
            self.__load_rqst_queue = self.__load_rqst_queue[nrqst:]
            return rqst if len(rqst) > 0 else None
        elif mode == 'response':
            if data is None:
                return
            data = [x[0] for x in data if x is not None]
            if len(data) == 0:
                return 
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
            
    
    def __min_unit(self, input):
        self.__min_val = np.min(np.vstack((np.min(np.array(input), 0), self.__min_val)), 0)
        self.__min_cnt += len(input)
        self.__min_state = False if self.__min_cnt < self.point_num else True
        
    
    def bin2dec(self, bin_arr, mode='little'):
        if mode == 'big':
            return np.dot(bin_arr, 2**np.arange(10)[::-1])  # [1,0,0,0] => 8
        elif mode == 'little':
            return np.dot(bin_arr, 2**np.arange(10))  # [1,0,0,0] => 1
    
    
    def dec2bin(self, dec, mode='little'):
        bin_arr = np.array(list(np.binary_repr(dec).zfill(10)))
        if mode == 'big':
            return bin_arr
        elif mode == 'little':
            return bin_arr[::-1]
    
    
    def detect_mcg_state(self):
        return self.mcg_state
    
    
    def get_morton_code(self):
        if self.mcg_state is True:
            return self.mc_decval    
        else:
            print('Err: MCG is not finished yet!')
            return None
        
    
    def cycle(self):
        read_rqst = []
        # receive read response from dram
        if len(self.dram_response) > 0:
            self.read_port('response', self.dram_response)
            self.dram_response = []
        
        # id generator --> x/y/z bit lane
        if len(self.__x_bit_lane) > 0:
            x_data = self.dec2bin(self.__x_bit_lane[0], mode='little')
            y_data = self.dec2bin(self.__y_bit_lane[0], mode='little')
            z_data = self.dec2bin(self.__z_bit_lane[0], mode='little')
            
            x_idx = np.arange(0, self.shifter_parallelism) * 3 + self.__bit_ptr
            y_idx = np.arange(0, self.shifter_parallelism) * 3 + self.__bit_ptr + 1
            z_idx = np.arange(0, self.shifter_parallelism) * 3 + self.__bit_ptr + 2
            
            x_idx = x_idx[x_idx < self.bits]
            y_idx = y_idx[y_idx < self.bits]
            z_idx = z_idx[z_idx < self.bits]
            self.__bit_ptr = (max(np.hstack((x_idx, y_idx, z_idx))) + 1) % self.bits
            
            self.__mc_buffer[self.__pts_ptr, x_idx] = x_data[x_idx//3]
            self.__mc_buffer[self.__pts_ptr, y_idx] = y_data[y_idx//3]
            self.__mc_buffer[self.__pts_ptr, z_idx] = z_data[z_idx//3]
            
            if self.__bit_ptr == 0:
                self.__x_bit_lane.pop(0)
                self.__y_bit_lane.pop(0)
                self.__z_bit_lane.pop(0)
                self.__pts_ptr += 1
            
            if self.__pts_ptr >= self.point_num:
                self.mc_decval = np.apply_along_axis(self.bin2dec, 1, self.__mc_buffer, mode='little')
                self.mcg_state = True 
                OUTPUT('[MCG] Morton code generator finished! clock: {}'.format(self.clock), devtype=DataSrc.morton_code_generator)
        
        # adder array --> div array (both in id generator)
        if len(self.__idgen_div_queue) > 0:
            ndat = min(self.idgen_parallelism, len(self.__idgen_div_queue))
            data = self.__idgen_div_queue[:ndat]
            self.__idgen_div_queue = self.__idgen_div_queue[ndat:]
            blk_idx = (np.array(data) / self.resolution).astype(np.int32)
            self.__x_bit_lane.extend(blk_idx[:, 0])
            self.__y_bit_lane.extend(blk_idx[:, 1])
            self.__z_bit_lane.extend(blk_idx[:, 2])
            
        # min unit --> id generator (adder array)
        if self.__min_state is True and len(self.__idgen_add_queue) > 0:
            ndat = min(self.idgen_parallelism, len(self.__idgen_add_queue))
            data = self.__idgen_add_queue[:ndat]
            self.__idgen_add_queue = self.__idgen_add_queue[ndat:]
            self.__idgen_div_queue.extend(data - self.__min_val)

        # dram --> min unit
        if len(self.__load_data_queue) > 0:
            if self.__min_state is False:
                ndat = min(self.min_parallelism, len(self.__load_data_queue))
                data = self.__load_data_queue[:ndat]
                self.__load_data_queue = self.__load_data_queue[ndat:]
                self.__min_unit(data)  
                    
        # send read requests to dram, to load point coordinats for min unit
        if len(self.__load_rqst_queue) > 0:
            read_rqst = self.read_port('request')
        
        # update clock
        if self.mcg_state is False:
            self.clock += 1
        
        return read_rqst if len(read_rqst) > 0 else None    
        

def gen_morton_code(xyz, resolution, bits=10):
    B, N, C = xyz.shape
    device = xyz.device
    
    xyz_min = xyz.min(1)[0]
    morton_code = torch.zeros(B, N, dtype=torch.long).to(device)
    cubic_idx = (xyz - xyz_min.view(B, 1, C)) / resolution
    cubic_idx = cubic_idx.long()
    # bitwise-interleave

    for i in range(bits):
        x_bit = (cubic_idx[:, :, 0] >> i) & 1
        y_bit = (cubic_idx[:, :, 1] >> i) & 1
        z_bit = (cubic_idx[:, :, 2] >> i) & 1
        morton_code |= (z_bit << (3 * i + 2)) + (y_bit << (3 * i + 1)) + (x_bit << (3 * i))

    return morton_code


def UnitTest():
    import data.data as data
    import dram
    sim = MortonCodeGenerator(data.xyz_1024, 0.0001)
    mem = dram.FixedLatencyDram()
    read_response = []
    while sim.detect_mcg_state() is False:
        read_request = sim.cycle(read_response)
        mem.push(read_request)
        read_response = mem.cycle()
    print('simulation result:', sim.mc_decval)
    
    # mc = gen_morton_code(data.xyz_1024.unsqueeze(0), 0.0001)
    # print('ground truth:', mc[0].cpu().numpy())
    
    # print('eval:', torch.equal(torch.tensor(sim.mc_decval), mc[0]))

import merge_sorter as sorter

class MortonCodeSorter:
    def __init__(self):
        self.__morton_code = []
        self.__sorter = sorter.MergeSorter()
        self.clock = 0
        self.state = True   # idle - True; busy - False
    
    
    def launch(self, morton_code):
        if len(self.__morton_code) == 0:
            self.__morton_code = morton_code
            self.state = False
        self.clock += 1
    
    
    def detect_mcs_state(self):
        return self.state
    
        
    def cycle(self):
        sorted_sequence, sorted_index = self.__sorter.cycle()
        
        if sorted_index is not None and self.state is False:
            self.state = True
            # OUTPUT('Merge Sort in finished, clock: {}'.format(self.clock))
            return sorted_index
        
        if len(self.__morton_code) > 0 and self.__sorter.get_state() is True:
            self.__sorter.launch(self.__morton_code)
            self.__morton_code = []
        
        if self.state is False:
            self.clock += 1
        
        return None
                    

if __name__ == '__main__':
    UnitTest()