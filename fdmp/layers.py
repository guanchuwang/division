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
from ops import fdmp_linear
from ops import fdmp_conv1d, fdmp_conv2d, fdmp_conv3d
from ops import fdmp_conv_transpose1d, fdmp_conv_transpose2d, fdmp_conv_transpose3d
from ops import fdmp_batch_norm1d, fdmp_batch_norm2d, fdmp_batch_norm3d

from ops_half import fdmp_linear_half
from ops_half import fdmp_conv1d_half, fdmp_conv2d_half, fdmp_conv3d_half
from ops_half import fdmp_conv_transpose1d_half, fdmp_conv_transpose2d_half, fdmp_conv_transpose3d_half

import cpp_extension.quantization as ext_quantization


class FDMP_Linear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, group=0):
        super(FDMP_Linear, self).__init__(input_features, output_features, bias)

    def forward(self, input):

        if not config.train:
            return super(FDMP_Linear, self).forward(input)

        if not config.half_precision: # full precision
            return self.forward_fp(input)

        else:
            return self.forward_half(input)

    def forward_fp(self, input):
        return fdmp_linear.apply(input, self.weight, self.bias)

    def forward_half(self, input):
        return fdmp_linear_half.apply(input, self.weight, self.bias)



class FDMP_Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(FDMP_Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if not config.train: # eval
            return super(FDMP_Conv1d, self).forward(input)

        # train
        if not config.half_precision: # full precision
            return self.forward_fp(input)

        else:
            return self.forward_half(input) # half precision

    def forward_fp(self, input):

        if self.padding_mode != 'zeros':
            return fdmp_conv1d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                     self.weight, self.bias, self.stride, _single(0), self.dilation, self.groups, self.scheme)

        return fdmp_conv1d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.scheme)

    def forward_half(self, input):

        if self.padding_mode != 'zeros':
            return fdmp_conv1d_half.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                          self.weight, self.bias, self.stride, _single(0), self.dilation, self.groups, self.scheme)

        return fdmp_conv1d_half.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.scheme)


class FDMP_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(FDMP_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if not config.train: # eval
            return super(FDMP_Conv2d, self).forward(input)

        # train
        if not config.half_precision: # full precision
            return self.forward_fp(input)

        else:
            return self.forward_half(input) # half precision

    def forward_fp(self, input):

        # train
        if self.padding_mode != 'zeros':
            return fdmp_conv2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return fdmp_conv2d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


    def forward_half(self, input):

        if self.padding_mode != 'zeros':
            return fdmp_conv2d_half.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return fdmp_conv2d_half.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FDMP_Conv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', group=0):
        super(QConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if not config.train: # eval
            return super(FDMP_Conv3d, self).forward(input)

        # train
        if not config.half_precision: # full precision
            return self.forward_fp(input)

        else:
            return self.forward_half(input) # half precision

    def forward_fp(self, input):
        # train
        if self.padding_mode != 'zeros':
            return fdmp_conv3d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                     self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return fdmp_conv3d.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward_half(self, input):

        if self.padding_mode != 'zeros':
            return fdmp_conv3d_half.apply(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return fdmp_conv3d_half.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class FDMP_BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(FDMP_BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if not config.train:
            return super(FDMP_BatchNorm1d, self).forward(input)

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
        return fdmp_batch_norm1d.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class FDMP_BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(FDMP_BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if not config.train:  # eval
            return super(FDMP_BatchNorm2d, self).forward(input)


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
        return fdmp_batch_norm2d.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class FDMP_BatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0):
        super(FDMP_BatchNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if not config.train: # eval
            return super(FDMP_BatchNorm3d, self).forward(input)

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
        return fdmp_batch_norm3d.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps, self.scheme)


class FDMP_ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(FDMP_ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)


    def forward(self, input, output_size=None):
        if not config.train:
            return super(FDMP_ConvTranspose1d, self).forward(input, output_size)


        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

        if not config.half_precision:
            return self.forward_fp(input, output_padding)

        else:
            return self.forward_half(input, output_padding)

    def forward_fp(self, input, output_padding):
        return fdmp_conv_transpose1d.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    def forward_half(self, input, output_padding):
        return fdmp_conv_transpose1d_half.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class FDMP_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(FDMP_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)


    def forward(self, input, output_size=None):
        if not config.train:
            return super(FDMP_ConvTranspose2d, self).forward(input, output_size)

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

        if not config.half_precision:
            return self.forward_fp(input, output_padding)

        else:
            return self.forward_half(input, output_padding)

    def forward_fp(self, input, output_padding):
        return fdmp_conv_transpose2d.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    def forward_half(self, input, output_padding):
        return fdmp_conv_transpose2d_half.apply(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class FDMP_ConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', group=0):
        super(FDMP_ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               padding, output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input, output_size=None):
        if not config.train:
            return super(FDMP_ConvTranspose3d, self).forward(input, output_size)

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore

        if not config.half_precision:
            return self.forward_fp(input, output_padding)

        else:
            return self.forward_half(input, output_padding)

    def forward_fp(self, input, output_padding):
        return fdmp_conv_transpose3d.apply(
                input, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

    def forward_half(self, input, output_padding):
        return fdmp_conv_transpose3d_half.apply(
                input, self.weight, self.bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)


class QReLU(nn.Module):
    def __init__(self, inplace=False):
        super(QReLU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ext_quantization.act_quantized_relu(input)


class QDropout(nn.Dropout):
    def __init__(self, p=0.5):
        super(QDropout, self).__init__(p=p)

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
        super(QMaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
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
        super(QAvgPool2d, self).__init__()
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