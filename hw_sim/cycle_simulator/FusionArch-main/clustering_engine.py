import torch
import pdb
import math
import dramAddrParse as mem_parser
import copy

class ClusteringEngine:
    def __init__(self, center_num, xyz, theshold=1e-4):
        self.ncenters = center_num
        self.ndc = center_num
        self.centroids = None
        self.pcd = xyz
        self.threshold = theshold   # temporarily un-used
        self.iteration = 0

        # self.io_traffic = 0 # i/o traffic between the clustering engine and global memory

        # local memory
        self.centers_idx = None
        self.centers = None    # centers coordinates, fixed on chip, each for a DC
        self.io_traffic = 3 * self.ncenters    # load (3-dim) centers coordinate from global memory
        self.cluster_result = [[] for _ in range(self.ncenters)]

        self.centroid_ptr = 0   # load centroids coordinates from the global memory, and broadcast to all DCs

        # intermediate queues 
        self.dcs2min = torch.ones(1 + self.ndc) * -1  # centroid idx (1) + dists to centers (ndc)
        self.red2div = torch.zeros(self.ndc, 3)     # reduction result

        # timing paramenters
        self.mean_unit_timers = [-1 for _ in range(self.ncenters)]

        # queue used to accept dram response
        self.dram_response = []
        
        # for response handling
        self.pending_read_centroid = []
        # self.pending_read_cluster = []  # load cluster points' coordinates from global memory
        
        # return to system
        # self.clustering_results = []
        # self.nonempty_cluster_num = 0
        self.equal_cluster_num = 0
        self.unchange_labels = [0 for _ in range(center_num)]
        self.has_been_update = [0 for _ in range(center_num)]
        self.is_idle = False
    
    def input(self, centroid_idx):
        if self.centroids is None:
            self.centroids = torch.tensor(centroid_idx)
            print('DBG: input centroids shape: {}'.format(self.centroids.shape))
            self.centers_idx = torch.index_select(self.centroids, 0, torch.randint(0, self.centroids.shape[0], (self.ncenters,)))
            print('initial centers_idx: {}'.format(self.centers_idx))
            self.centers = torch.index_select(self.pcd, 0, self.centers_idx)
            self.is_idle = False
            
            self.centroid_ptr = 0   # reset centroid pointer
            self.mean_unit_timers = [-1 for _ in range(self.ncenters)]
            self.cluster_result = [[] for _ in range(self.ncenters)]
            self.unchange_labels = [0 for _ in range(self.ncenters)]
            self.has_been_update = [0 for _ in range(self.ncenters)]

    def cycle(self):
        # is_idle = False
        ret = None
        # clustering_results = None
        requests_list = []

        # DRAM response handler (for DC)
        read_response = [x[0] for x in self.dram_response if x[1] == False]
        for res in read_response:
            res = mem_parser.parse_addr2pidx(res)
            for i in range(len(self.pending_read_centroid)):
                if self.pending_read_centroid[i][0] == res and self.pending_read_centroid[i][1] > 0:
                    self.pending_read_centroid[i][1] -= 1
                    if self.pending_read_centroid[i][1] == 0:
                        self.pending_read_centroid.remove([res, 0])
                        # self.dram_response.remove((res, False))
                        # print('[Read Response] received centroid: {} from DRAM'.format(res))
                        centroid = self.pcd[res]  # load centroid coordinates from global memory
                        self.dcs2min[0] = res   # record centroid index at 0
                        self.io_traffic += 3
                        for i in range(self.ndc):
                            center = self.centers[i]    # each center is fixed on a DC
                            self.dcs2min[i+1] = torch.sum((centroid - center) ** 2, dim=-1)
                    break

        # MEAN Unit - division phase
        if (torch.count_nonzero(self.red2div, dim=1) > 0).nonzero().numel() > 0:
            nonzero_idx = (self.red2div!=0).nonzero()[0, 0].item()
            old_center = copy.deepcopy(self.centers[nonzero_idx])
            self.centers[nonzero_idx] = self.red2div[nonzero_idx]   # update center coordinates
            self.has_been_update[nonzero_idx] = 1
            # print('update center {} to {}'.format(nonzero_idx, self.red2div[nonzero_idx]))
      
            if old_center.equal(self.centers[nonzero_idx]): # center unchanged, terminate the clustering
                self.unchange_labels[nonzero_idx] = 1
                # print('center {} unchanged, self.red2div: {}'.format(nonzero_idx, self.red2div))
            if self.unchange_labels.count(1) == self.ncenters - self.cluster_result.count([]):
                self.is_idle = True
                # print('[Send Request] write clustering results to DRAM in batch')
                ret = copy.deepcopy(self.cluster_result)
                self.centroids = None   # reset centroids
                self.centers_idx = None # reset centers_idx
                self.centers = None     # reset centers
                print('[CLU Report] Iteration {} finished'.format(self.iteration+1))
                self.iteration = 0
                
            self.red2div[nonzero_idx] = torch.zeros(3)   # reset reduction result

            # if self.nonempty_cluster_num == self.ncenters - self.cluster_result.count([]):
            if self.has_been_update.count(1) == self.ncenters - self.cluster_result.count([]) and self.is_idle is False:
                # current iteration finished
                self.mean_unit_timers = [-1 for _ in range(self.ncenters)]
                self.centroid_ptr = 0   # reset centroid pointer
                self.iteration += 1
                self.cluster_result = [[] for _ in range(self.ncenters)]
                # self.equal_cluster_num = 0
                self.unchange_labels = [0 for _ in range(self.ncenters)]
                self.has_been_update = [0 for _ in range(self.ncenters)]
                # self.nonempty_cluster_num = 0
                print('[CLU Report] Iteration {} finished'.format(self.iteration))

        # MEAN Unit - reduction phase
        if self.centroids is not None and self.centroid_ptr >= self.centroids.shape[0] and len(self.pending_read_centroid) == 0 and (self.dcs2min == -1).nonzero().numel() == self.ndc + 1:
            for i in range(self.ncenters):
                if self.has_been_update[i] == 0:
                    if self.mean_unit_timers[i] == -1 and len(self.cluster_result[i]) > 0:
                        self.mean_unit_timers[i] = math.ceil(math.log2(len(self.cluster_result[i])))
                    
                    if self.mean_unit_timers[i] == 0:
                        # Assume the coordinates have been hold in the Clustering Engine
                        self.red2div[i] = torch.mean(torch.index_select(self.pcd, 0, torch.tensor(self.cluster_result[i])), dim=0)
                        break
                    elif self.mean_unit_timers[i] > 0:
                        self.mean_unit_timers[i] -= 1
                        break

        # MIN Unit
        if self.dcs2min[0] >= 0:
            min_dist, min_idx = torch.min(self.dcs2min[1:], 0)
            self.cluster_result[min_idx].append(int(self.dcs2min[0].item()))
            # print('reduce centroid {} to cluster {}, centroid coordinate: {}'.format(int(self.dcs2min[0].item()), min_idx, self.pcd[int(self.dcs2min[0].item())]))
            self.dcs2min = torch.ones(self.ndc+1) * -1 # reset intermediate queue 'dcs2min'
        
        # Distance Calculator
        if self.centroids is not None and self.centroid_ptr < self.centroids.shape[0]:
            addr_x, addr_y, addr_z = mem_parser.gen_xyzcoord_addr(self.centroids[self.centroid_ptr])
            requests_list.append([addr_x, 0, 2])
            requests_list.append([addr_y, 0, 2])
            requests_list.append([addr_z, 0, 2])
            self.pending_read_centroid.append([self.centroids[self.centroid_ptr], 3])
            # print('[Send Request] send a read request of centroid {}'.format(self.centroids[self.centroid_ptr]))

            self.centroid_ptr += 1


        return self.is_idle, ret, requests_list
    

if __name__ == '__main__':
    center_num = 8
    # centroid_idx = torch.tensor([ 50,   3, 122,  60, 466,  28,   8,   7, 417, 381, 101, 422,
    #     242, 371, 477,  90,   1, 295,  95, 338,  19, 266,  11,  87,
    #     233,  22, 383, 171, 114, 271, 315, 144, 209,  29, 151, 184,
    #     333,  77, 156, 253, 365, 361, 229, 469, 507, 275,  71,  61,
    #      56,  44,  94, 364,  53,  20, 428, 400, 106, 382, 455, 441,
    #     468, 326,  93, 208, 452, 270,  65, 121, 245, 458,  68, 292,
    #      70, 136, 359, 354, 113, 161, 325, 471, 481, 385, 425, 123,
    #      88, 470,  46,  14, 236,  85,  63, 498, 504, 254,  35, 268,
    #     357,  37, 260, 148,  89, 320, 476,  59, 162, 257, 135, 492,
    #     508, 377,  24, 310,  39, 111, 222, 283, 376, 312, 124,  23,
    #     132, 438, 327,  91, 139, 309, 337, 307], dtype=torch.long)
    import data
    xyz = data.xyz_1024
    # centroid_idx = torch.tensor([444, 1, 328, 3, 23, 744, 362, 6, 5, 245, 10, 11, 700, 533, 322, 13, 773, 18, 16, 139, 342, 105, 20, 85, 457, 36, 55, 673, 28, 800, 370, 57, 21, 173, 47, 33, 591, 891, 512, 98, 757, 40, 808, 137, 295, 727, 499, 97, 246, 50, 51, 634, 642, 71, 355, 191, 822, 353, 1016, 64, 1001, 950, 306, 666, 522, 872, 72, 997, 394, 672, 644, 972, 827, 494, 89, 545, 65, 169, 819, 594, 86, 449, 898, 611, 677, 341, 835, 209, 321, 1005, 37, 761, 94, 765, 324, 406, 83, 871, 780, 874, 752, 125, 1002, 234, 299, 31, 415, 107, 730, 208, 540, 616, 132, 943, 116, 152, 119, 548, 696, 378, 128, 123, 127, 111, 665, 133, 766, 683, 138, 58, 121, 560, 459, 669, 776, 555, 896, 155, 149, 54, 161, 751, 500, 336, 875, 828, 201, 779, 777, 853, 712, 461, 495, 165, 745, 479, 30, 483, 782, 181, 148, 549, 451, 846, 747, 940, 951, 187, 189, 1015, 704, 1014, 237, 1018, 859, 850, 178, 734, 895, 117, 933, 778, 477, 998, 541, 816, 758, 231, 593, 538, 396, 67, 185, 215, 280, 547, 225, 952, 446, 1022, 227, 603, 797, 840, 248, 247, 731, 383, 556, 369, 48, 239, 509, 175, 153, 388, 202, 511, 243, 352, 251, 233, 590, 1011, 252, 26, 43, 101, 481, 261, 219, 262, 429, 259, 386, 999, 258, 925, 274, 275, 520, 784, 277, 157, 831, 820, 14, 377, 626, 551, 249, 297, 301, 936, 302, 303, 158, 543, 615, 304, 675, 311, 312, 639, 900, 320, 637, 316, 382, 892, 2, 77, 329, 330, 331, 332, 78, 276, 360, 195, 334, 335, 427, 435, 317, 102, 343, 460, 344, 485, 583, 906, 159, 351, 786, 455, 93, 671, 379, 537, 361, 92, 1023, 498, 365, 618, 905, 238, 134, 668, 171, 387, 349, 380, 289, 260, 385, 458, 571, 697, 432, 568, 472, 346, 392, 217, 391, 340, 199, 443, 76, 395, 144, 845, 399, 282, 180, 211, 715, 407, 402, 867, 403, 438, 404, 425, 408, 604, 410, 911, 411, 135, 497, 445, 888, 63, 423, 532, 420, 305, 857, 49, 426, 759, 588, 654, 775, 431, 434, 573, 193, 436, 437, 80, 147, 32, 374, 771, 441, 452, 830, 783, 151, 701, 447, 140, 450, 418, 228, 114, 236, 296, 596, 486, 463, 416, 174, 954, 465, 466, 468, 470, 471, 232, 474, 476, 338, 478, 91, 493, 480, 955, 557, 482, 381, 489, 142, 518, 339, 255, 641, 974, 197, 490, 492, 279, 179, 693, 143, 501, 456, 319, 688, 502, 563, 525, 503, 292, 504, 505, 172, 519, 508, 507, 9, 510, 46, 176, 985, 514, 364, 575, 167, 516, 517, 897, 163, 904, 126, 524, 390, 908, 326, 582, 861, 735, 240, 528, 226, 530, 454, 534, 473, 212, 373, 224, 542, 544, 558, 414, 52, 550, 75, 552, 553, 88, 19, 401, 829, 130, 562, 264, 629, 559, 564, 453, 565, 367, 546, 566, 286, 539, 154, 569, 570, 12, 635, 572, 284, 574, 421, 577])
    centroid_idx = torch.tensor([563, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 13, 13, 14, 49, 700, 16, 990, 18, 19, 20, 21, 281, 23, 546, 85, 27, 28, 29, 30, 918, 31, 898, 33, 113, 35, 460, 36, 39, 40, 571, 41, 877, 42, 503, 45, 251, 46, 47, 48, 50, 51, 52, 54, 279, 56, 58, 60, 176, 61, 799, 679, 63, 64, 65, 773, 59, 240, 72, 73, 703, 75, 76, 77, 315, 78, 936, 80, 83, 84, 22, 86, 87, 89, 90, 69, 683, 91, 545, 92, 93, 329, 456, 94, 95, 96, 192, 1015, 101, 102, 382, 604, 103, 105, 24, 110, 395, 265, 800, 444, 530, 647, 938, 116, 117, 702, 119, 129, 121, 120, 123, 124, 125, 127, 443, 128, 521, 130, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 145, 457, 975, 177, 704, 150, 597, 159, 155, 156, 304, 157, 158, 173, 160, 327, 555, 163, 677, 166, 167, 169, 526, 170, 171, 174, 175, 216, 605, 643, 179, 126, 180, 181, 798, 182, 183, 974, 186, 188, 189, 190, 191, 193, 808, 194, 195, 778, 197, 678, 840, 861, 201, 202, 203, 205, 504, 986, 220, 468, 211, 774, 213, 219, 941, 214, 215, 217, 222, 224, 225, 226, 227, 477, 228, 229, 231, 319, 863, 982, 383, 185, 234, 235, 232, 238, 62, 239, 210, 243, 246, 962, 212, 248, 154, 252, 886, 254, 255, 257, 259, 950, 260, 261, 264, 266, 820, 429, 268, 488, 424, 269, 273, 991, 274, 250, 275, 276, 783, 277, 280, 34, 282, 283, 287, 288, 290, 293, 249, 296, 297, 298, 810, 299, 300, 302, 303, 37, 153, 305, 306, 787, 309, 310, 759, 139, 311, 420, 697, 312, 727, 313, 314, 317, 321, 322, 323, 165, 358, 328, 43, 330, 331, 332, 333, 334, 897, 335, 336, 337, 340, 341, 342, 256, 343, 97, 979, 432, 346, 805, 347, 348, 349, 351, 352, 354, 355, 356, 463, 245, 359, 360, 380, 361, 362, 363, 365, 366, 367, 368, 369, 370, 409, 371, 374, 375, 377, 378, 469, 0, 381, 385, 344, 388, 389, 390, 391, 199, 392, 393, 394, 396, 845, 398, 399, 38, 541, 401, 402, 403, 404, 405, 406, 408, 187, 172, 711, 410, 411, 413, 415, 416, 431, 417, 445, 418, 325, 419, 284, 111, 421, 423, 425, 426, 427, 428, 558, 292, 430, 82, 198, 433, 434, 436, 437, 438, 439, 440, 654, 441, 442, 55, 151, 446, 447, 178, 448, 449, 450, 499, 451, 452, 453, 454, 455, 972, 146, 236, 458, 668, 459, 462, 461, 464, 70, 954, 465, 466, 467, 470, 149, 625, 471, 472, 473, 474, 475, 476, 338, 478, 479, 480, 482, 483, 484, 208, 485, 486, 376, 715, 489, 490, 106, 492, 493, 495, 497, 498, 500, 501, 108, 502, 57, 745, 100, 233, 295, 505, 168, 508, 509, 510, 512, 514, 516, 517, 518, 519, 520, 143, 522, 523, 524, 301, 525, 326, 543, 864, 207, 98, 527, 652, 529, 531, 507, 532, 533, 535, 829, 537])
    
    ce = ClusteringEngine(center_num, xyz)
    ce.input(centroid_idx[:128])

    while True:
        state, ret, requests = ce.cycle()

        # send requests to dram
        while len(requests) > 0:
            req = requests.pop(0)
            ce.dram_response.append((int(req[0], 16), False))
    
        if state is True and ret is not None:
            print('clustering finished, ret = ', ret)
            break