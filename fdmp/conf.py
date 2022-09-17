import ast
import os
import warnings
import json

class QuantizationConfig:
    def __init__(self):

        self.compress_bn_input = True
        self.num_classes = 1000
        self.non_quant = False
        self.max_thread = 1024

        self.simulate = False
        self.group_size = 256
        self.lfc_block = 8
        self.hfc_bit_num = 2
        self.half_precision = False

        # Ablation study
        self.lfc_flag = True
        self.hfc_flag = True

        # Memory management flag
        self.empty_cache_threshold = None
        self.pipeline_threshold = None
        self.cudnn_benchmark_conv2d = True
        self.swap = False

        # Debug related flag
        self.debug_memory_model = ast.literal_eval(os.environ.get('DEBUG_MEM', "False"))
        self.debug_speed = ast.literal_eval(os.environ.get('DEBUG_SPEED', "False"))
        self.debug_memory_op_forward = False # True #
        self.debug_memory_op_backward = False # True #
        self.debug_remove_bn = False
        self.debug_remove_relu = False
        self.debug_fd_memory = False

    def __str__(self):
        return json.dumps(self.__dict__)


def config_init(args):

    config.simulate = args.simulate
    config.group_size = args.group_size
    config.lfc_block = args.lfc_block
    config.hfc_bit_num = args.hfc_bit_num
    config.half_precision = args.amp
    config.rm_lfc = args.rm_lfc
    config.rm_hfc = args.rm_hfc
    config.debug_fd_memory = args.debug_fd_memory

    if args.gpu_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in args.gpu_devices])

    return



config = QuantizationConfig()

