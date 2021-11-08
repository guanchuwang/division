from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch import Tensor, device, dtype

from layers import FDMP_Linear
from layers import FDMP_Conv1d, FDMP_Conv2d, FDMP_Conv3d
from layers import FDMP_BatchNorm1d, FDMP_BatchNorm2d, FDMP_BatchNorm3d
from layers import FDMP_ConvTranspose1d, FDMP_ConvTranspose2d, FDMP_ConvTranspose3d
from layers import QReLU, QDropout, QMaxPool2d, QAvgPool2d
from conf import config
# from torch.cuda.amp import autocast # os.environ["CUDA_VISIBLE_DEVICES"] should before this import command

class FDMP_Module(nn.Module):
    layer_num = 0
    layer_counter = 0

    def __init__(self, model):
        super(FDMP_Module, self).__init__()
        self.model = model
        FDMP_Module.convert_layers(self.model)
        FDMP_Module.restore_layers(self.model) # restore the last linear layer
        print(self.model)
        # hegsns

    @staticmethod
    def convert_layers(module):
        for name, child in module.named_children():
            # Do not convert layers that are already converted
            # if isinstance(child, (QConv1d, QConv2d, QConv3d, QConvTranspose1d, QConvTranspose2d, QConvTranspose3d,
            #                       QBatchNorm1d, QBatchNorm2d, QBatchNorm3d, QSyncBatchNorm,
            #                       QReLU, QDropout, QLinear, QMaxPool2d, QAvgPool2d)):
            #     continue

            if isinstance(child, (FDMP_Linear,
                                  FDMP_Conv1d, FDMP_Conv2d, FDMP_Conv3d,
                                  FDMP_BatchNorm1d, FDMP_BatchNorm2d, FDMP_BatchNorm3d,
                                  FDMP_ConvTranspose1d, FDMP_ConvTranspose2d, FDMP_ConvTranspose3d,
                                  QReLU, QDropout, QMaxPool2d, QAvgPool2d)):

                continue

            if isinstance(child, nn.Conv1d):
                setattr(module, name, FDMP_Conv1d(child.in_channels, child.out_channels,
                                                  child.kernel_size, child.stride, child.padding, child.dilation,
                                                  child.groups, child.bias is not None, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.Conv2d):
                setattr(module, name, FDMP_Conv2d(child.in_channels, child.out_channels,
                                                  child.kernel_size, child.stride, child.padding, child.dilation,
                                                  child.groups, child.bias is not None, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.Conv3d):
                setattr(module, name, FDMP_Conv3d(child.in_channels, child.out_channels,
                                                  child.kernel_size, child.stride, child.padding, child.dilation,
                                                  child.groups, child.bias is not None, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.ConvTranspose1d):
                setattr(module, name, FDMP_ConvTranspose1d(child.in_channels, child.out_channels,
                                                           child.kernel_size, child.stride, child.padding, child.output_padding,
                                                           child.groups, child.bias, child.dilation, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.ConvTranspose2d):
                setattr(module, name, FDMP_ConvTranspose2d(child.in_channels, child.out_channels,
                                                           child.kernel_size, child.stride, child.padding, child.output_padding,
                                                           child.groups, child.bias, child.dilation, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.ConvTranspose3d):
                setattr(module, name, FDMP_ConvTranspose3d(child.in_channels, child.out_channels,
                                                           child.kernel_size, child.stride, child.padding, child.output_padding,
                                                           child.groups, child.bias, child.dilation, child.padding_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.BatchNorm1d) and config.compress_bn_input:
                setattr(module, name, FDMP_BatchNorm2d(child.num_features, child.eps, child.momentum,
                                                       child.affine, child.track_running_stats))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.BatchNorm2d) and config.compress_bn_input:
                setattr(module, name, FDMP_BatchNorm2d(child.num_features, child.eps, child.momentum,
                                                       child.affine, child.track_running_stats))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.BatchNorm3d) and config.compress_bn_input:
                setattr(module, name, FDMP_BatchNorm3d(child.num_features, child.eps, child.momentum,
                                                       child.affine, child.track_running_stats))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.ReLU):
                setattr(module, name, QReLU())
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.Dropout):
                setattr(module, name, QDropout(child.p))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.MaxPool2d):
                setattr(module, name, QMaxPool2d(child.kernel_size, child.stride,
                                                 child.padding, child.dilation, child.return_indices, child.ceil_mode))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.AvgPool2d):
                setattr(module, name, QAvgPool2d(child.kernel_size, child.stride, child.padding,
                                                 child.ceil_mode, child.count_include_pad, child.divisor_override))
                FDMP_Module.layer_num += 1

            elif isinstance(child, nn.Linear):
                setattr(module, name, FDMP_Linear(child.in_features, child.out_features,
                                                  child.bias is not None))
                FDMP_Module.layer_num += 1

            FDMP_Module.convert_layers(child)

    @staticmethod
    def restore_layers(module):

        for name, child in module.named_children():

            if FDMP_Module.layer_counter == FDMP_Module.layer_num:
                return

            elif FDMP_Module.layer_counter == FDMP_Module.layer_num-1:

                if isinstance(child, FDMP_Linear):

                    setattr(module, name, nn.Linear(child.in_features, child.out_features, child.bias is not None))
                    FDMP_Module.layer_counter += 1
                    return

            elif isinstance(child, (FDMP_Linear,
                            FDMP_Conv1d, FDMP_Conv2d,
                            FDMP_BatchNorm1d, FDMP_BatchNorm2d,
                            FDMP_ConvTranspose1d, FDMP_ConvTranspose2d,
                            QReLU, QDropout, QMaxPool2d, QAvgPool2d)):

                FDMP_Module.layer_counter += 1
                continue

            FDMP_Module.restore_layers(child)


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        config.train = mode
        return super(FDMP_Module, self).train(mode)

    def eval(self):
        config.train = False
        return super(FDMP_Module, self).eval()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super(FDMP_Module, self).load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(FDMP_Module, self).state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret

    # def train(self, mode=True):
    #     config.mode = 0 if mode == True else 1
    #     return super(FDMP_Module, self).train(mode)
    #
    # def eval(self):
    #     config.mode = 1
    #     return super(FDMP_Module, self).eval()
    #
    # def prep(self, input_shape):
    #     super(FDMP_Module, self).eval()
    #     config.mode = 2
    #     prob_x = torch.ones(input_shape)
    #     self.model(prob_x)
    #     return self.train()  # default as train state

    # def cuda(self, device=None):
    #     FDMP_Module.matrix_cuda(self.model, device)
    #     return super(FDMP_Module, self).cuda(device)
    #
    # @staticmethod
    # def matrix_cuda(module, device=None):
    #
    #     for name, child in module.named_children():
    #         if hasattr(child, "dct_matrix"):
    #             child.dct_matrix.cuda(device)
    #             print(child, device, child.dct_matrix.device)
    #         else:
    #             FDMP_Module.matrix_cuda(child, device)


