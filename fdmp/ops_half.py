import numpy as np
import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import cpp_extension.backward_func as ext_backward_func
from conf import config
import time

from utils.actnn_utils import *

from ops import fdmp_linear
from ops import fdmp_conv2d, fdmp_conv1d
from ops import fdmp_conv_transpose1d, fdmp_conv_transpose2d

from conf import config
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import custom_fwd, custom_bwd


class fdmp_linear_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        return fdmp_linear.forward(ctx, input, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_linear.backward(ctx, grad_output)


class fdmp_conv1d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return fdmp_conv1d.forward(ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv1d.backward(ctx, grad_output)


class fdmp_conv2d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return fdmp_conv2d.forward(ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv2d.backward(ctx, grad_output)


class fdmp_conv3d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return fdmp_conv3d.forward(ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv3d.backward(ctx, grad_output)


class fdmp_conv_transpose1d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transpose1d.forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv_transpose1d.backward(ctx, grad_output)


class fdmp_conv_transpose2d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transpose2d.forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv_transpose2d.backward(ctx, grad_output)


class fdmp_conv_transpose3d_half(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transpose3d.forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return fdmp_conv_transpose3d.backward(ctx, grad_output)

