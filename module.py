from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch import Tensor, device, dtype

from layers import MDCT_Conv2d, MDCT_BatchNorm2d, QReLU # MDCT_Conv1d,

from conf import config
from torch.cuda.amp import autocast # os.environ["CUDA_VISIBLE_DEVICES"] should before this import command

class MDCT_Module(nn.Module):

    def __init__(self, model):
        super(MDCT_Module, self).__init__()
        self.model = model
        MDCT_Module.convert_layers(self.model)

    # @staticmethod
    # def update_conv_window_size(module, window_size=1., hfc_bit_num=2):
    #
    #     for name, child in module.named_children():
    #         # Do not convert layers that are already quantized
    #         if isinstance(child, (MDCT_Conv2d)):
    #             child.window_size = window_size
    #             child.hfc_bit_num = hfc_bit_num
    #         else:
    #             MDCT_Module.update_conv_window_size(child, window_size, hfc_bit_num)
    #
    # @staticmethod
    # def update_bn_window_size(module, window_size=1., hfc_bit_num=2):
    #
    #     for name, child in module.named_children():
    #         # Do not convert layers that are already quantized
    #         if isinstance(child, MDCT_BatchNorm2d):
    #             child.window_size = window_size
    #             child.hfc_bit_num = hfc_bit_num
    #         else:
    #             MDCT_Module.update_bn_window_size(child, window_size, hfc_bit_num)

    @staticmethod
    def convert_layers(module):
        for name, child in module.named_children():
            # Do not convert layers that are already quantized
            # if isinstance(child, (QConv1d, QConv2d, QConv3d, QConvTranspose1d, QConvTranspose2d, QConvTranspose3d,
            #                       QBatchNorm1d, QBatchNorm2d, QBatchNorm3d, QSyncBatchNorm,
            #                       QReLU, QDropout, QLinear, QMaxPool2d, QAvgPool2d)):
            #     continue

            if isinstance(child, (MDCT_Conv2d, MDCT_BatchNorm2d)):

                continue
            # if isinstance(child, nn.Conv1d):
            #     setattr(module, name, MDCT_Conv1d(child.in_channels, child.out_channels,
            #         child.kernel_size, child.stride, child.padding, child.dilation,
            #         child.groups, child.bias is not None, child.padding_mode))
            if isinstance(child, nn.Conv2d):
                setattr(module, name, MDCT_Conv2d(child.in_channels, child.out_channels,
                    child.kernel_size, child.stride, child.padding, child.dilation,
                    child.groups, child.bias is not None, child.padding_mode))
            # elif isinstance(child, nn.Conv3d):
            #     setattr(module, name, QConv3d(child.in_channels, child.out_channels,
            #         child.kernel_size, child.stride, child.padding, child.dilation,
            #         child.groups, child.bias is not None, child.padding_mode))
            # elif isinstance(child, nn.ConvTranspose1d):
            #     setattr(module, name, QConvTranspose1d(child.in_channels, child.out_channels,
            #         child.kernel_size, child.stride, child.padding, child.output_padding,
            #         child.groups, child.bias, child.dilation, child.padding_mode))
            # elif isinstance(child, nn.ConvTranspose2d):
            #     setattr(module, name, QConvTranspose2d(child.in_channels, child.out_channels,
            #         child.kernel_size, child.stride, child.padding, child.output_padding,
            #         child.groups, child.bias, child.dilation, child.padding_mode))
            # elif isinstance(child, nn.ConvTranspose3d):
            #     setattr(module, name, QConvTranspose3d(child.in_channels, child.out_channels,
            #         child.kernel_size, child.stride, child.padding, child.output_padding,
            #         child.groups, child.bias, child.dilation, child.padding_mode))
            # elif isinstance(child, nn.BatchNorm1d) and config.enable_quantized_bn:
            #     setattr(module, name, QBatchNorm1d(child.num_features, child.eps, child.momentum,
            #         child.affine, child.track_running_stats))
            elif isinstance(child, nn.BatchNorm2d) and config.compress_bn_input:
                setattr(module, name, MDCT_BatchNorm2d(child.num_features, child.eps, child.momentum,
                    child.affine, child.track_running_stats))
            # elif isinstance(child, nn.BatchNorm3d) and config.enable_quantized_bn:
            #     setattr(module, name, QBatchNorm3d(child.num_features, child.eps, child.momentum,
            #         child.affine, child.track_running_stats))
            # elif isinstance(child, nn.Linear):
            #     setattr(module, name, QLinear(child.in_features, child.out_features,
            #         child.bias is not None))
            elif isinstance(child, nn.ReLU):
                setattr(module, name, QReLU())
            # elif isinstance(child, nn.Dropout):
            #     setattr(module, name, QDropout(child.p))
            # elif isinstance(child, nn.MaxPool2d):
            #     setattr(module, name, QMaxPool2d(child.kernel_size, child.stride,
            #         child.padding, child.dilation, child.return_indices, child.ceil_mode))
            # elif isinstance(child, nn.AvgPool2d):
            #     setattr(module, name, QAvgPool2d(child.kernel_size, child.stride, child.padding,
            #         child.ceil_mode, child.count_include_pad, child.divisor_override))
            elif isinstance(child, nn.Sequential):
                # print(child)
                MDCT_Module.convert_layers(child)
            else:
                MDCT_Module.convert_layers(child)

    # @autocast()
    def forward(self, *args, **kwargs):
        # with autocast():
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        config.train = mode
        return super(MDCT_Module, self).train(mode)

    def eval(self):
        config.train = False
        return super(MDCT_Module, self).eval()

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        # remove the prefix "model." added by this wrapper
        new_state_dict = OrderedDict([("model." + k,  v) for k, v in state_dict.items()])
        return super(MDCT_Module, self).load_state_dict(new_state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(MDCT_Module, self).state_dict(destination, prefix, keep_vars)

        # remove the prefix "model." added by this wrapper
        ret = OrderedDict([(k[6:], v) for k, v in ret.items()])
        return ret

    # def train(self, mode=True):
    #     config.mode = 0 if mode == True else 1
    #     return super(MDCT_Module, self).train(mode)
    #
    # def eval(self):
    #     config.mode = 1
    #     return super(MDCT_Module, self).eval()
    #
    # def prep(self, input_shape):
    #     super(MDCT_Module, self).eval()
    #     config.mode = 2
    #     prob_x = torch.ones(input_shape)
    #     self.model(prob_x)
    #     return self.train()  # default as train state

    # def cuda(self, device=None):
    #     MDCT_Module.matrix_cuda(self.model, device)
    #     return super(MDCT_Module, self).cuda(device)
    #
    # @staticmethod
    # def matrix_cuda(module, device=None):
    #
    #     for name, child in module.named_children():
    #         if hasattr(child, "dct_matrix"):
    #             child.dct_matrix.cuda(device)
    #             print(child, device, child.dct_matrix.device)
    #         else:
    #             MDCT_Module.matrix_cuda(child, device)


