import pdb
import math

def gen_addr(row, col, bank, bank_group, rank, channel=0):
    row <<= 18
    rank <<= 17
    bank <<= 15
    bank_group <<= 13
    col <<= 6
    
    addr = row | rank | bank | bank_group | col
    return hex(addr)

def regen_addr(addr):
    col = (addr >> 6) & 127
    bank_group = (addr >> 13) & 3
    bank = (addr >> 15) & 3
    rank = (addr >> 17) & 1
    row = (addr >> 18) & 65535
    channel = 0

    return row, col, bank, bank_group, rank, channel

def dtype(addr):
    row, col, bank, bank_group, rank, channel = regen_addr(addr)
    typ = -1
    if rank == 0:
        if bank_group < 3:
            typ = 0 # point cloud coordinates
        else:
            if bank < 2:
                typ = 1 # centroid indices
            else:
                typ = 2 # cluster indices
    else:
        if bank_group < 2:
            typ = 3 # neighbors indices
        else:
            typ = 4 # feature vectors
    return typ

# parse point cloud index to dram address
def gen_xyzcoord_addr(point_idx):
    # assert point_idx < 1024 and point_idx >= 0
    bank = point_idx >> 7
    row = point_idx >> 9
    col = point_idx - (bank << 7)
    rank = 0
    channel = 0
    
    addr_x = gen_addr(row, col, bank, 0, rank, channel)
    addr_y = gen_addr(row, col, bank, 1, rank, channel)
    addr_z = gen_addr(row, col, bank, 2, rank, channel)

    return addr_x, addr_y, addr_z   # a list of point_idx's x,y,z coordinates' dram address

# parse dram address (1-dim) to point cloud index
def parse_addr2pidx(addr):
    row, col, bank, bank_group, rank, channel = regen_addr(addr)
    point_idx = (row << 9) + (bank << 7) + col
    return point_idx  # may be point_idx or point_idx+1

# parse centroid index to dram address
def gen_centroids_addr(point_idx, all_centroids):
    bank_group = 3
    rank = 0
    channel = 0
    idx = all_centroids.index(point_idx)
    bank = int(format(idx, '08b')[-8], 2)   # idx % 256 // 128
    row = idx >> 8
    col = idx - (bank << 7) - (row << 8)
    addr = gen_addr(row, col, bank, bank_group, rank, channel)
    return addr

# idx is not point index, but the index of all_centroids
def gen_centroids_addr_withidx(idx):
    bank_group = 3
    rank = 0
    channel = 0
    bank = int(format(idx, '08b')[-8], 2)   # idx % 256 // 128
    row = idx >> 8
    col = idx - (bank << 7) - (row << 8)
    addr = gen_addr(row, col, bank, bank_group, rank, channel)
    return addr

def gen_centroids_addr_inbatch(centroid_set):
    rank = 0
    channel = 0
    bank_group = 3
    
    request_num = math.ceil(len(centroid_set) / 8)
    addrs = []
    for i in range(request_num):
        bank = int(format(i, '06b')[-5], 2)
        row = i >> 5
        col = ((i << 6) - (row << 11) - (bank << 10)) >> 3
        addrs.append(gen_addr(row, col, bank, bank_group, rank, channel))
    return addrs

# parse dram address to centroid index
def parse_addr2centroidsidx(addr, all_centroids):
    row, col, bank, bank_group, rank, channel = regen_addr(addr)
    idx = (row << 8) + (bank << 7) + col
    point_idx = all_centroids[idx]
    return point_idx

def gen_cluster_addr(cluster_size):
    channel = 0
    rank = 0
    bank_group = 3
    bank = 2
    row = cluster_size // 256
    col = 0
    request_num = math.ceil(cluster_size / 8)

    addrs = []
    for _ in range(request_num):
        addrs.append(gen_addr(row, col%128, bank, bank_group, rank, channel))
        col += 8
        bank = int(format(col, '08b')[-8], 2) + 2
    
    return addrs

# parse centroid index to its neighbors' dram address
def gen_neighbors_addr(centroid_idx, all_centroids):
    idx = all_centroids.index(centroid_idx)
    rank = 1
    channel = 0
    bank_group = int(format(idx,'08b')[-5], 2)  # idx * 256 % (2**16) // (2**15)
    bank = int(format(idx, '04b')[-4:-2], 2)
    row = idx >> 5
    col = int(format(idx, '04b')[-2:], 2) << 5
    addr = gen_addr(row, col, bank, bank_group, rank, channel)
    addr1 = gen_addr(row, col+8, bank, bank_group, rank, channel)
    addr2 = gen_addr(row, col+16, bank, bank_group, rank, channel)
    addr3 = gen_addr(row, col+24, bank, bank_group, rank, channel)
    
    return addr, addr1, addr2, addr3

def gen_neighbors_addr_withidx(idx):
    rank = 1
    channel = 0
    bank_group = int(format(idx,'08b')[-5], 2)  # idx * 256 % (2**16) // (2**15)
    bank = int(format(idx, '04b')[-4:-2], 2)
    row = idx >> 5
    col = int(format(idx, '04b')[-2:], 2) << 5
    addr = gen_addr(row, col, bank, bank_group, rank, channel)
    addr1 = gen_addr(row, col+8, bank, bank_group, rank, channel)
    addr2 = gen_addr(row, col+16, bank, bank_group, rank, channel)
    addr3 = gen_addr(row, col+24, bank, bank_group, rank, channel)
    
    return addr, addr1, addr2, addr3

def gen_neighbors_addr_inbatch(neighbors_set):
    channel = 0
    rank = 1

    addrs = []
    request_num = math.ceil(sum([len(x) for x in neighbors_set]) / 8)
    for i in range(request_num):
        row = i >> 7
        bank_group = ((i << 6) - (row << 13)) >> 12
        bank = ((i << 6) - (row << 13) - (bank_group << 12)) >> 10
        col = ((i << 6) - (row << 13) - (bank_group << 12) - (bank << 10)) >> 3
        addrs.append(gen_addr(row, col, bank, bank_group, rank, channel))
    
    return addrs

# parse dram address to neighbor index
def parse_addr2neighborsidx(addr, all_centroids):
    row, col, bank, bank_group, rank, channel = regen_addr(addr)
    idx = row*32 + bank_group * 16 + bank * 4 + col // 32
    
    return all_centroids[idx]

def gen_feature_addr(point_idx, feature_dim):
    point_idx = int(point_idx)
    footprint = point_idx * feature_dim * 8
    rank = 1
    channel = 0
    row = footprint >> 13
    # import math
    # bank_group = int(format(point_idx, '09b')[-(9 - int(math.log2(feature_dim)) + 1)], 2)
    # print('dbg: fetch pos:',-(7 - math.ceil(math.log2(feature_dim))+1))
    # bank = int(format(point_idx, '08b')[-(7 - int(math.log2(feature_dim)))-2:-(7 - int(math.log2(feature_dim)))], 2)
    bank_group = (footprint - (row << 13)) >> 12
    bank = (footprint - (row << 13) - (bank_group << 12)) >> 10
    col = (footprint - (row << 13) - (bank_group << 12) - (bank << 10)) >> 3

    request_num = math.ceil(feature_dim / 8)
    addrs = []
    for i in range(request_num):
        addrs.append(gen_addr(row, col + i*8, bank, bank_group+2, rank, channel))
    # print('[+] idx: {}, dim: {}, row: {}, col: {}, bank: {}, bank_group: {}, addrs: {}'.format(point_idx, feature_dim, row, col, bank, bank_group, addrs))

    return addrs

def parse_addr2featureidx(addr, feature_dim):
    row, col, bank, bank_group, rank, channel = regen_addr(addr)
    bank_group -= 2
    idx = ((row << 10) + (bank_group << 9) + (bank << 7) + col) // feature_dim
    # print('[-] addr: {}, row: {}, col: {}, bank: {}, bank_group: {}, idx: {}'.format(addr, row, col, bank, bank_group, idx))
    
    return idx

if __name__ == '__main__':
    # addr1 = gen_addr(0,0,0,0,0,0)
    # addr2 = gen_addr(0,1,0,0,0,0)
    # addr3 = gen_addr(0,2,0,0,0,0)
    # addr4 = gen_addr(0,3,0,0,0,0)
    # import libdramsim3 as dram
    # # print('addr:', addr1, addr2, addr3)
    # addr = [addr1, addr2, addr3, addr4]
    # # dram.add_pending(int(addr1, 16), False, 0)
    # # dram.add_pending(int(addr2, 16), False,0)
    # # dram.add_pending(int(addr3, 16),False,0)
    # # dram.add_pending(int(addr4, 16),False,0)
    # count = 0
    # while True:
    #     if len(addr) > 0:
    #         dram.add_pending(int(addr.pop(0), 16), False, 0)
    #     ret = dram.cycle()
    #     if ret is not None and len(ret) > 0:
    #         count += 1
    #         if count == 4:
    #             break
    # import torch
    # from 0 to 1024, step is 2
    # centroids = torch.arange(0, 1023, 2).tolist() # from 0 to 1024, step is 2
    pointlist = [0, 41, 42, 125, 127]
    fdims = 3
    for pidx in pointlist:
        addrs = gen_feature_addr(pidx, fdims)
        for addr in addrs:
            parse_addr2featureidx(int(addr, 16), fdims)
    

    # addr0, addr1, addr2, addr3 = gen_neighbors_addr(2, centroids)
    # print('Test2:', parse_addr2neighborsidx(int(addr0, 16), centroids)) 
    # print('Test2:', parse_addr2neighborsidx(int(addr1, 16), centroids)) 
    # print('Test2:', parse_addr2neighborsidx(int(addr2, 16), centroids)) 
    # print('Test2:', parse_addr2neighborsidx(int(addr3, 16), centroids)) 

    # addr0, addr1, addr2, addr3 = gen_neighbors_addr(256, centroids)
    # print('Test256:', parse_addr2neighborsidx(int(addr0, 16), centroids)) 
    # print('Test256:', parse_addr2neighborsidx(int(addr1, 16), centroids)) 
    # print('Test256:', parse_addr2neighborsidx(int(addr2, 16), centroids)) 
    # print('Test256:', parse_addr2neighborsidx(int(addr3, 16), centroids)) 

    # addr0, addr1, addr2, addr3 = gen_neighbors_addr(1022, centroids)
    # print('Test1022:', parse_addr2neighborsidx(int(addr0, 16), centroids)) 
    # print('Test1022:', parse_addr2neighborsidx(int(addr1, 16), centroids)) 
    # print('Test1022:', parse_addr2neighborsidx(int(addr2, 16), centroids)) 
    # print('Test1022:', parse_addr2neighborsidx(int(addr3, 16), centroids)) 



    # addr_group0 = gen_neighbors_addr(2, centroids)
    # print('Test2:', parse_addr2neighborsidx(int(addr_group0, 16), centroids))


    # addr_group0 = gen_neighbors_addr(256, centroids)
    # print('Test256:', parse_addr2neighborsidx(int(addr_group0, 16), centroids))

    # addr_group0 = gen_neighbors_addr(1022, centroids)
    # print('Test1022:', parse_addr2neighborsidx(int(addr_group0, 16), centroids))
