# import numpy as np
import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
# import exact.cpp_extension.backward_func as ext_backward_func
import cpp_extension.backward_func as ext_backward_func
from conf import config
import time

from utils import *

from fdmp import FDMP, WDCT
from conf import config, QuantizationConfig
# from torch.cuda.amp import autocast as autocast
# from torch.cuda.amp import custom_fwd, custom_bwd

conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0


class fdmp_linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):

        feature_pack = FDMP.fdmp(1, input, WDCT.dct_1d, config.conv_window_size, weight.device)

        empty_cache(config.empty_cache_threshold)

        # ctx.scheme = scheme
        ctx.saved = feature_pack, weight, bias
        ctx.other_args = input.shape

        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # if ctx.scheme:
        #     ctx.scheme.set_scale(grad_output)

        feature_pack, weight, bias = ctx.saved
        input_shape = ctx.other_args

        input = FDMP.de_fdmp(1, feature_pack, WDCT.dct_1d, input_shape, config.conv_window_size, weight.device)
        del feature_pack, ctx.saved

        empty_cache(config.empty_cache_threshold)

        # TODO: the following implementation might not be optimal
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        # rank = len(grad_output.shape)

        grad_output_flatten = grad_output.view(-1, C_out)
        input_flatten = input.view(-1, C_in)
        # print(grad_output_flatten.shape, weight.shape)
        grad_input = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)

        # grad_input = grad_output.mm(weight)
        # grad_weight = grad_output.t().mm(input)
        if bias is not None:
            # grad_bias = grad_output.sum(0)
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None



class fdmp_convnd(Function):
    @staticmethod
    def run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        # if config.simulate:
        #     feature_pack = FDMP.fdmp_simulation(input)
        # else:
        #     feature_pack = FDMP.fdmp(input)

        feature_pack = FDMP.fdmp(input)


        ## Save variable for backward
        # ctx.scheme = scheme
        ctx.saved = feature_pack, weight, bias
        ctx.other_args = input.shape, stride, padding, dilation, groups

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv%dd forward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes(feature_pack)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        # print("layer:", n)
        # print(input.dtype, weight.dtype)
        # print(input.device, weight.device)
        # print(input.shape, weight.shape)
        # print(bias, stride, padding, dilation, groups)
        # import pdb
        # pdb.set_trace()

        return forward_op(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def run_backward(n, ctx, grad_output, bias_reduce_dims, aug):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return None, None, None, None, None, None, None, None

        input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        stride = aug(stride)
        dilation = aug(dilation)

        feature_pack, weight, bias = ctx.saved
        # idct_api = WDCT.idct_1d if n == 1 else WDCT.idct_2d

        if config.simulate:
            input = FDMP.de_fdmp_simulation(feature_pack, input_shape)

        else:
            input = FDMP.de_fdmp(feature_pack, input_shape)

        del feature_pack, ctx.saved
        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd backward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        # print(input.dtype)
        # print(grad_output.dtype)
        # print(weight.dtype)
        # print(bias)
        # print(ctx.needs_input_grad[0], ctx.needs_input_grad[1])

        use_pipeline = False
        if config.pipeline_threshold:
            ws_mem = compute_tensor_bytes([grad_output, input, input])
            if (ws_mem > config.pipeline_threshold and
                ctx.needs_input_grad[1] and ctx.needs_input_grad[0]):
                use_pipeline = True

        if use_pipeline:
            micro_batch_size = (ws_mem + config.pipeline_threshold) // config.pipeline_threshold
            raw_input = input
            raw_grad_output = grad_output
            input = torch.chunk(input, micro_batch_size)
            grad_output = torch.chunk(grad_output,  micro_batch_size)
            grad_weight = None

            for i in range(micro_batch_size):
                input[i][:], grad_weight_tmp = ext_backward_func.cudnn_convolution_backward(
                        input[i], grad_output[i], weight, padding, stride, dilation, groups,
                        config.cudnn_benchmark_conv2d, False, False,
                        [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
                if grad_weight is None:
                    grad_weight = grad_weight_tmp
                else:
                    grad_weight += grad_weight_tmp
            grad_input = raw_input
            grad_output = raw_grad_output
        else:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_backward(
                input, grad_output, weight, padding, stride, dilation, groups,
                config.cudnn_benchmark_conv2d, False, False,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(bias_reduce_dims)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class fdmp_conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return convnd.run_forward(1, F.conv1d, ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        return convnd.run_backward(1, ctx, grad_output, [0, 2], _single)


class fdmp_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return fdmp_convnd.run_forward(2, F.conv2d, ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_convnd.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class fdmp_conv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return fdmp_convnd.run_forward(3, F.conv3d, ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_convnd.run_backward(3, ctx, grad_output, [0, 2, 3, 4], _triple)



class fdmp_batch_norm_nd(Function):
    @staticmethod
    def run_forward(n, ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]


        if config.simulate:
            feature_pack = FDMP.fdmp_simulation(input)

        else:
            feature_pack = FDMP.fdmp(input)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global bn_layer_ct, total_act_mem
            print("========== bn forward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1
            total_act_mem += compute_tensor_bytes(feature_pack)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        ## Save variable for backward
        # ctx.scheme = scheme
        ctx.saved = (feature_pack, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)
        ctx.other_args = input.shape

        return output

    @staticmethod
    def run_backward(n, ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None

        feature_pack, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved
        input_shape = ctx.other_args

        if config.simulate:
            input = FDMP.de_fdmp_simulation(feature_pack, input_shape)

        else:
            input = FDMP.de_fdmp(feature_pack, input_shape)

        del feature_pack, ctx.saved
        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        # print(input.dtype)
        # print(grad_output.dtype)
        # print(weight.dtype)

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        return grad_input, None, None, grad_weight, grad_bias, None, None, None


class fdmp_batch_norm1d(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps):
        return fdmp_batch_norm_nd.run_forward(1, ctx, input, running_mean, running_var, weight, bias,
                                              training, exponential_average_factor, eps)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_batch_norm_nd.run_backward(1, ctx, grad_output)


class fdmp_batch_norm2d(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps):
        return fdmp_batch_norm_nd.run_forward(2, ctx, input, running_mean, running_var, weight, bias,
                                              training, exponential_average_factor, eps)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_batch_norm_nd.run_backward(2, ctx, grad_output)


class fdmp_batch_norm3d(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps):
        return fdmp_batch_norm_nd.run_forward(3, ctx, input, running_mean, running_var, weight, bias,
                                              training, exponential_average_factor, eps)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_batch_norm_nd.run_backward(3, ctx, grad_output)


class fdmp_conv_transposend(Function):
    @staticmethod
    def run_forward(n, forward_op, dct_op, ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):

        feature_pack = FDMP.fdmp(n, input, dct_op, config.conv_window_size, weight.device)

        # ctx.scheme = scheme
        ctx.saved = feature_pack, weight, bias
        ctx.other_args = (input.shape, stride, padding, output_padding, dilation, groups)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv%dd_transpose forward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes(quantized)
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        return forward_op(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def run_backward(n, dct_op, ctx, grad_output, bias_reduce_dims, aug):
        # if ctx.scheme:
        #     ctx.scheme.set_scale(grad_output)

        input_shape, stride, padding, output_padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        output_padding = aug(output_padding)
        stride = aug(stride)
        dilation = aug(dilation)

        feature_pack, weight, bias = ctx.saved
        # idct_api = WDCT.idct_1d if n == 1 else WDCT.idct_2d
        input = FDMP.de_fdmp(n, feature_pack, dct_op, input_shape, config.conv_window_size, weight.device)

        del feature_pack, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd_transpose backward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        use_pipeline = False
        if config.pipeline_threshold:
            ws_mem = compute_tensor_bytes([grad_output, input, input])
            if (ws_mem > config.pipeline_threshold and
                ctx.needs_input_grad[1] and ctx.needs_input_grad[0]):
                use_pipeline = True

        if use_pipeline:
            micro_batch_size = (ws_mem + config.pipeline_threshold) // config.pipeline_threshold
            raw_input = input
            raw_grad_output = grad_output
            input = torch.chunk(input, micro_batch_size)
            grad_output = torch.chunk(grad_output,  micro_batch_size)
            grad_weight = None

            for i in range(micro_batch_size):
                input[i][:], grad_weight_tmp = ext_backward_func.cudnn_convolution_transpose_backward(
                        input[i], grad_output[i], weight, padding, output_padding, stride, dilation, groups,
                        config.cudnn_benchmark_conv2d, False, False,
                        [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])
                if grad_weight is None:
                    grad_weight = grad_weight_tmp
                else:
                    grad_weight += grad_weight_tmp
            grad_input = raw_input
            grad_output = raw_grad_output
        else:
            grad_input, grad_weight = ext_backward_func.cudnn_convolution_transpose_backward(
                input, grad_output, weight, padding, output_padding, stride, dilation, groups,
                config.cudnn_benchmark_conv2d, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(bias_reduce_dims)
        else:
            grad_bias = None

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class fdmp_conv_transpose1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transposend.run_forward(1, F.conv_transpose1d, WDCT.dct_1d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_conv_transposend.run_backward(1, WDCT.dct_1d, ctx, grad_output, [0, 2], _single)


class fdmp_conv_transpose2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transposend.run_forward(2, F.conv_transpose2d, WDCT.dct_2d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_conv_transposend.run_backward(2, WDCT.dct_2d, ctx, grad_output, [0, 2, 3], _pair)


class fdmp_conv_transpose3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        return fdmp_conv_transposend.run_forward(3, F.conv_transpose3d, WDCT.dct_3d, ctx, input, weight, bias, stride,
                                            padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        return fdmp_conv_transposend.run_backward(3, WDCT.dct_3d, ctx, grad_output, [0, 2, 3, 4], _triple)
