import pdb
import torch

class FixedLatencyDram:
    class DramEntry:
        def __init__(self, request):
            data, byte_size, data_src, is_write = request
            self.request = (data, data_src, is_write)
            self.timer = 50*(byte_size//64) if byte_size > 64 else 50
            # self.timer = 1  # for debug

    def __init__(self):
        self.pending_list = []
    
    def cycle(self):            
        return_list = []
        for i in range(len(self.pending_list)):
            self.pending_list[i].timer -= 1
            if self.pending_list[i].timer <= 0:
                return_list.append(self.pending_list[i].request)
            
        # pop the request from the pending list when its timer is 0
        self.pending_list = list(filter(lambda req:req.timer>0, self.pending_list))

        return return_list

    def push(self, requests):
        if isinstance(requests, list):
            for rqst in requests:
                self.pending_list.append(self.DramEntry(rqst))
        else:
            self.pending_list.append(self.DramEntry(requests))

    def debug_show_dram(self):
        output = {}
        for req in self.pending_list:
            output[req.request[0]] = req.timer
        print(output)


from enum import Enum
class DataSrc(Enum):
    fps_unit = 0
    knn_unit = 1
    morton_code_generator = 2
    feature_aggregation_unit = 3
    

class IOType(Enum):
    request = 0
    response = 1


def isin(val, list):
    val_type = type(val[0])
    for item in list:
        item_type = type(item[0])
        if val_type == item_type:
            if val_type == torch.Tensor:
                if torch.equal(val[0], item[0]) and val[1] == item[1] and val[2] == item[2]:
                    return True
            else:
                if val[0] == item[0] and val[1] == item[1] and val[2] == item[2]:
                    return True
    return False

class OnChipBuffer:
    def __init__(self):
        self.coordinate_buffer = []
    
    # handle requests before dram
    def request(self, rqst):
        ret = []
        for r in rqst:
            if r[1] == DataSrc.fps_unit or r[1] == DataSrc.knn_unit:
                if r[0] in self.coordinate_buffer:
                    ret.append(r)
            elif r[1] == DataSrc.morton_code_generator:
                ret.append(r)
        return ret
    
    # receive responses from dram
    def response(self, resp):
        if isinstance(resp, list):
            for r in resp:
                if r[1] == DataSrc.fps_unit or r[1] == DataSrc.knn_unit or r[1] == DataSrc.morton_code_generator:
                    self.coordinate_buffer.append(r[0])
        else:
            if resp[1] == DataSrc.fps_unit or resp[1] == DataSrc.knn_unit or resp[1] == DataSrc.morton_code_generator:
                self.coordinate_buffer.append(resp[0])