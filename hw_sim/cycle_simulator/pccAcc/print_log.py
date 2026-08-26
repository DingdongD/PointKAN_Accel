from enum import Enum
from configparser import ConfigParser
import os

class DeviceType(Enum):
    fps_unit = 0
    knn_unit = 1
    morton_code_generator = 2
    sorter = 3
    quant_prec_generator = 4
    quant_unit = 5
    dequant_unit = 6
    feature_decomposer = 7
    feature_aggregation_unit = 8
    feature_computation_unit = 9
    system = 10


active_dt = [DeviceType.system]

config = ConfigParser()
# config_path = './config/pointnet2_cls_x16.ini'
# config_path = './config/dgcnn_cls_x16.ini'
# config_path = './config/pointnet2_ps_x16.ini'
# config_path = './config/pointnet2_s_x32.ini'
# config_path = './config/dgcnn_ps_x64.ini'
config_path = './config/dgcnn_s_x16.ini'
# config_path = './config/demo.ini'
config.read(config_path)
filename = config.get('LOG', 'filename')


# active_dt = [DeviceType.fps_unit, DeviceType.knn_unit, DeviceType.morton_code_generator, DeviceType.sorter, DeviceType.quant_prec_generator, DeviceType.quant_unit, DeviceType.dequant_unit, DeviceType.feature_decomposer, DeviceType.feature_aggregation_unit, DeviceType.feature_computation_unit, DeviceType.system]

def OUTPUT(str, devtype=None, mode='F'):
    if devtype is None or devtype in active_dt:
        # mode is tereminal (T)
        if mode == 'T':
            print(str)
        # mode is file (F)
        elif mode == 'F':
            with open(filename, 'a') as f:
                f.writelines(str + '\n')
        # no output (N)
        elif mode == 'N':
            return
        else:
            print('Err: invalid mode of output!')
            exit(1)   