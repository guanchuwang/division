# The code is compatible with PyTorch 1.6/1.7
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch import Tensor
from torch.nn.modules.pooling import _size_2_t, _single, _pair, _triple, _MaxPoolNd, _AvgPoolNd

# from actnn.qscheme import QScheme
# from actnn.qbnscheme import QBNScheme
from conf import config
from ops import mdct_conv2d, mdct_batch_norm

# from actnn.ops import linear, batch_norm, conv1d, conv2d, conv3d, sync_batch_norm
# from actnn.ops import conv_transpose1d, conv_transpose2d, conv_transpose3d
# import actnn.cpp_extension.quantization as ext_quantization
import cpp_extension.quantization as ext_quantization


class MDCT_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(MDCT_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)

        # self.scheme = None
        # self.window_size = 1.
        # self.hfc_bit_num = 2
        # self.dct_matrix = None

    def forward(self, input):

        if not config.train: # eval
            return super(MDCT_Conv2d, self).forward(input)

        # train
        if self.padding_mode != 'zeros':
            return mdct_conv2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        ## Remove Prep stage ###
        # elif config.mode == 2: # prep
        #     # self.dct_matrix = MDCT_op.generate_dct_matrix(input.shape[-1], round(self.window_size*input.shape[-1] + 0.5))
        #       return super(MDCT_Conv2d, self).forward(input)

        return mdct_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MDCT_BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(MDCT_BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        # self.scheme = None
        # self.window_size = 1.
        # self.hfc_bit_num = 2
        # self.dct_matrix = None

    def forward(self, input):
        if not config.train:  # eval
            return super(MDCT_BatchNorm2d, self).forward(input)


        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return mdct_batch_norm.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

        ### Remove Prep Stage ###
        # elif config.mode == 2:  # prep
        #     # self.dct_matrix = MDCT_op.generate_dct_matrix(input.shape[-1], round(self.window_size*input.shape[-1] + 0.5))
        #     return super(MDCT_BatchNorm2d, self).forward(input)


class QReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_relu(input)


class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return ext_quantization.act_quantized_dropout(input, self.p)
        else:
            return super(QDropout, self).forward(input)


class QMaxPool2d(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, input):
        return ext_quantization.act_quantized_max_pool2d(
            input, self.kernel_size, self.stride,
            self.padding, self.dilation, self.ceil_mode,
            self.return_indices)


class QAvgPool2d(_AvgPoolNd):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: bool = None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if (stride is not None) else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        # TODO: implement memory-optimized cuda kernel for this.
        #return F.avg_pool2d(input, self.kernel_size, self.stride,
        #                    self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)
        warnings.warn("avg_pool2d is replcaed by max_pool2d, because the optimized cuda kernel"
                      "for avg_pool2d is not implemented.")
        return ext_quantization.act_quantized_max_pool2d(
            input, self.kernel_size, self.stride,
            self.padding, (1, 1), self.ceil_mode,
            False)