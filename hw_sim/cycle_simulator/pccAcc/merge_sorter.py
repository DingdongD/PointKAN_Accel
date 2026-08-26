import numpy as np
import torch
import pdb
import math
import sys
from print_log import OUTPUT
from print_log import DeviceType

class MergeSorter:
    def __init__(self):
        self.__data_rate = 1.97
        self.__latency = sys.maxsize
        self.__sorted_sequence = None
        self.__sorted_index = None
        self.clock = 0
        self.state = True   # idle - True; busy - False
        self.pending_queue = []
    
    def launch(self, input, dim=0):
        if self.state is True:
            self.__sorted_sequence = np.sort(input, dim)
            self.__sorted_index = np.argsort(input, dim)
            self.__latency = math.ceil(len(input) / self.__data_rate)
            self.clock += 1
            self.state = False
            OUTPUT('[Sorter] Launch in working queue', devtype=DeviceType.sorter)
        else:
            self.pending_queue.append((input, dim))
            OUTPUT('[Sorter] Launch in pending queue', devtype=DeviceType.sorter)
    
    def get_state(self):
        return self.state
        
    def cycle(self):
        if len(self.pending_queue) > 0 and self.state is True:
            input, dim = self.pending_queue.pop(0)
            self.launch(input, dim)
            OUTPUT('[Sorter] Working queue <-- pending queue, clock: {}'.format(self.clock), devtype=DeviceType.sorter)
        
        if self.clock >= self.__latency:
            sorted_sequence = self.__sorted_sequence
            sorted_index = self.__sorted_index
            self.__sorted_sequence = None
            self.__sorted_index = None
            self.state = True
            OUTPUT('[Sorter] Merge Sort is finished, clock: {}'.format(self.clock), devtype=DeviceType.sorter)
            self.clock = 0
            return (sorted_sequence, sorted_index)
        else:
            if self.__sorted_sequence is not None:
                self.clock += 1
        
        return (None, None)


