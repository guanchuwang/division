import numpy as np
import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import cpp_extension.backward_func as ext_backward_func
from conf import config
import time
from torch.cuda.amp import autocast as autocast

from utils.actnn_utils import *
# from utils.mdct_utils import *
from timer import global_timer
# from dct_matrix import generate_dct_matrix, dct_2d, zero_padding

from fdmp import MDCT_op, FDMP
from conf import config, QuantizationConfig

conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0


class mdct_convnd(Function):
    @staticmethod
    def run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        # t0 = time.time()
        # dct_matrix = MDCT_op.generate_dct_matrix2(input.shape[-1], round(config.conv_window_size*input.shape[-1] + 0.5), weight.device)
        # torch.cuda.synchronize()
        # t1 = time.time()
        feature_pack = FDMP.fdmp(input, config.conv_window_size, weight.device)
        # print(t1 - t0)

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

        return forward_op(input, weight, bias, stride, padding, dilation, groups)
        # return forward_op(x_lfc, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def run_backward(n, ctx, grad_output, bias_reduce_dims, aug):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return None, None, None, None, None, None, None, None
        # if ctx.scheme:
        #     ctx.scheme.set_scale(grad_output)

        input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        stride = aug(stride)
        dilation = aug(dilation)

        feature_pack, weight, bias = ctx.saved
        # time1 = time.time()
        # dct_matrix = dct_matrix_buf[input_shape[-1]]
        # dct_matrix = MDCT_op.generate_dct_matrix2(input_shape[-1], round(config.conv_window_size*input_shape[-1] + 0.5), weight.device)
        input = FDMP.de_fdmp(feature_pack, input_shape, config.conv_window_size, weight.device)
        # torch.cuda.synchronize()
        # global_timer.run(time.time() - time1, 'dct')

        del feature_pack, ctx.saved
        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd backward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            conv2d_layer_ct += 1
            print("WS: %.2f MB" % (compute_tensor_bytes([grad_output, input, input]) / 1024 ** 2))

        # input = input.to(torch.float)
        print(input.dtype)
        print(grad_output.dtype)
        print(weight.dtype)
        print(bias)
        print(ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        # hegsns

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

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None


class mdct_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return mdct_convnd.run_forward(2, F.conv2d, ctx, input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        return mdct_convnd.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class mdct_batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]

        # time1 = time.time()
        # dct_matrix = dct_matrix_buf[input.shape[-1]]
        # dct_matrix = MDCT_op.generate_dct_matrix2(input.shape[-1], round(config.bn_window_size*input.shape[-1] + 0.5), weight.device)
        feature_pack = FDMP.fdmp(input, config.bn_window_size, weight.device)
        # torch.cuda.synchronize()
        # global_timer.run(time.time() - time1, 'dct')

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
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None

        feature_pack, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved
        input_shape = ctx.other_args

        # time1 = time.time()
        # dct_matrix = dct_matrix_buf[input_shape[-1]]
        # dct_matrix = MDCT_op.generate_dct_matrix2(input_shape[-1], round(config.bn_window_size*input_shape[-1] + 0.5), weight.device)
        input = FDMP.de_fdmp(feature_pack, input_shape, config.bn_window_size, weight.device)
        # torch.cuda.synchronize()
        # global_timer.run(time.time() - time1, 'dct')

        del feature_pack, ctx.saved
        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        print(input.dtype)
        print(grad_output.dtype)
        print(weight.dtype)

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        # if ctx.scheme:
        #     ctx.scheme.if_allocate_perlayer()
        return grad_input, None, None, grad_weight, grad_bias, None, None, None


