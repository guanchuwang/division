import numpy as np
import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import cpp_extension.backward_func as ext_backward_func
from conf import config
import time

import torch_1_8_0_dct as torch_dct
# from pythran.mdct_utils import auto_window, encode, decode, fd_feature

from utils.actnn_utils import *
# from utils.mdct_utils import *


conv2d_layer_ct = 0
bn_layer_ct = 0
total_act_mem = 0


def fdq(x, window_size, quant_bit=2):

    if window_size >= 1:
        return x
    elif window_size <= 0:
        return torch.zeros_like(x.shape).cuda()

    x_dct = mask_dct_2d(x, window_size)
    x_idct = mask_idct_2d(x_dct)
    x_hfc = x - x_idct  # high frequency component
    x_min_group = x_hfc.min(dim=2)[0].min(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
    x_max_group = x_hfc.max(dim=2)[0].max(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
    quant_step = (x_max_group - x_min_group) / (2 ** quant_bit) + 1e-8
    x_reference = x_min_group # + quant_step/2
    x_hfc_quant = torch.round((x_hfc - x_reference) / quant_step)  # .type(torch.int)
    x_hfc_quant = torch.clamp(x_hfc_quant, min=0, max=2 ** quant_bit - 1)
    x_hfc_dequant = x_hfc_quant * quant_step + x_min_group
    x_recover = x_idct + x_hfc_dequant

    return x_recover


def mask_dct(x, window_size):
    x_dct = torch_dct.dct(x, norm='ortho')
    m1 = round(window_size * x_dct.shape[2] + 0.5)
    x_dct[:, :, m1:] = 0
    return x_dct


def mask_dct_2d(x, window_size):
    x_dct = torch_dct.dct_2d(x, norm='ortho')

    m1 = round(window_size * x_dct.shape[2] + 0.5)
    m2 = round(window_size * x_dct.shape[3] + 0.5)

    x_dct[:, :, m1:, m2:] = 0
    return x_dct

    # return x_dct[:, :, 0:m1, 0:m2].clone(), \
    #        x_dct[:, :, m1:, 0:m2].mean(dim=(2, 3)).unsqueeze(dim=2).unsqueeze(dim=3).clone(), \
    #        x_dct[:, :, 0:m1, m2:].mean(dim=(2, 3)).unsqueeze(dim=2).unsqueeze(dim=3).clone(), \
    #        x_dct[:, :, m1:, m2:].mean(dim=(2, 3)).unsqueeze(dim=2).unsqueeze(dim=3).clone()

def mask_idct(x_dct):
    x_idct = torch_dct.idct(x_dct, norm='ortho')
    return x_idct

def mask_idct_2d(x_dct):
    x_idct = torch_dct.idct_2d(x_dct, norm='ortho')
    return x_idct


# def auto_mask_dct_2d(x, window_size):
#     start_time = time.time()
#
#     x_dct = torch_dct.dct_2d(x, norm='ortho')
#     x_dct = x_dct.detach().cpu().numpy()
#
#     pixel = 0.1
#     x_dct_feature = fd_feature(x_dct, pixel=pixel)
#
#     time1 = time.time()
#     print("--- fd_feature {} seconds ---".format(time1 - start_time))
#
#     batch_window_size, _ = auto_window(x_dct_feature, window_size, pixel=pixel, search_st_window_index=3)
#
#     print(batch_window_size)
#     print(batch_window_size.mean())
#     time2 = time.time()
#     print("--- Search window_size {} seconds ---".format(time2 - time1))
#
#     x_lfc_dct, x_hfc_ave1, x_hfc_ave2, x_hfc_ave3, m1, m2 = encode(x_dct, batch_window_size)
#
#     time3 = time.time()
#     print("--- Encode featuremap {} seconds ---".format(time3 - time2))
#
#     x_lfc_dct  = torch.from_numpy(x_lfc_dct) .type(torch.float).cuda()
#     x_hfc_ave1 = torch.from_numpy(x_hfc_ave1).type(torch.float).cuda()
#     x_hfc_ave2 = torch.from_numpy(x_hfc_ave2).type(torch.float).cuda()
#     x_hfc_ave3 = torch.from_numpy(x_hfc_ave3).type(torch.float).cuda()
#     m1         = torch.from_numpy(m1)        .type(torch.int).cuda()
#     m2         = torch.from_numpy(m2)        .type(torch.int).cuda()
#
#     time4 = time.time()
#     print("--- Forward {} seconds ---".format(time4 - start_time))
#
#     return x_lfc_dct, x_hfc_ave1, x_hfc_ave2, x_hfc_ave3, m1, m2


class mdct_convnd(Function):
    @staticmethod
    def run_forward(n, forward_op, ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None,
                    window_size=1., hfc_bit_num=2):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        # Fixed window_size
        # x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave = mask_dct_2d(input, window_size)
        # x_dct = mask_dct_2d(input, window_size)

        x_recover = fdq(input, window_size, hfc_bit_num)

        # Auto window_size
        # start_time = time.time()
        # x_mask_dct = encoder(input, window_size, barrier_num, max_search_time, min_window_size)
        # print("--- Forward {} seconds ---".format(time.time() - start_time))


        ## Save variable for backward
        ctx.scheme = scheme
        # ctx.save_for_backward(x_mask_dct, weight, bias)
        # ctx.save_for_backward(x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, m1, m2, weight, bias)
        # ctx.saved = x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, weight, bias
        # ctx.saved = x_dct, weight, bias
        ctx.saved = x_recover, weight, bias
        ctx.other_args = (input.shape, stride, padding, dilation, groups)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global conv2d_layer_ct, total_act_mem
            print("========== conv%dd forward %d ==========" % (n, conv2d_layer_ct))
            get_memory_usage(True)
            if input.shape[1] == 3:
                conv2d_layer_ct = 0
            conv2d_layer_ct += 1
            total_act_mem += compute_tensor_bytes([x_mask_dct])
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        return forward_op(input, weight, bias, stride, padding, dilation, groups)
        # return forward_op(x_lfc, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def run_backward(n, ctx, grad_output, bias_reduce_dims, aug):
        # if not ctx.needs_input_grad[1]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        #     return None, None, None, None, None, None, None, None
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)

        input_shape, stride, padding, dilation, groups = ctx.other_args
        padding = aug(padding)
        stride = aug(stride)
        dilation = aug(dilation)

        # Fixed window size
        # x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, weight, bias = ctx.saved
        # x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, m1, m2, weight, bias = ctx.saved_variables
        # input = auto_mask_idct_2d(x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, m1, m2, input_shape)
        # x_mask_dct, weight, bias = ctx.saved_variables
        # x_dct, weight, bias = ctx.saved
        input, weight, bias = ctx.saved

        # Fixed window size
        # input = mask_idct_2d(x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, input_shape)
        # input = mask_idct_2d(x_dct)

        # Auto window size
        # input = decoder(x_mask_dct)
        # del x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, ctx.saved

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global conv2d_layer_ct
            print("========== conv%dd backward %d ==========" % (n, conv2d_layer_ct))
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

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, \
               None, None


class mdct_conv1d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None,
                    window_size=1., hfc_bit_num=2):
        return mdct_convnd.run_forward(1, F.conv1d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme,
                                       window_size, hfc_bit_num)

    @staticmethod
    def backward(ctx, grad_output):
        return mdct_convnd.run_backward(1, ctx, grad_output, [0, 2], _single)


class mdct_conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, scheme=None,
                    window_size=1., hfc_bit_num=2):
        return mdct_convnd.run_forward(2, F.conv2d, ctx, input, weight, bias, stride, padding, dilation, groups, scheme,
                                       window_size, hfc_bit_num)

    @staticmethod
    def backward(ctx, grad_output):
        return mdct_convnd.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class mdct_batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps, scheme,
                window_size=1., hfc_bit_num=2):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return ext_backward_func.cudnn_batch_norm(
        #         input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)[0]

        x_recover = fdq(input, window_size, hfc_bit_num)

        # Fixed window_size
        # x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave = mask_dct_2d(input, window_size)


        # Search window_size
        # x_mask_dct = encoder(input, window_size, barrier_num, max_search_time, min_window_size)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_forward:
            global bn_layer_ct, total_act_mem
            print("========== bn forward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            if input.shape[1] == 3:
                bn_layer_ct = 0
            bn_layer_ct += 1
            total_act_mem += compute_tensor_bytes([x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave])
            total_act_mem += compute_tensor_bytes([x_mask_dct])
            print("Act mem: %.2f MB" % (total_act_mem / 1024 ** 2))

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        ## Save variable for backward
        ctx.scheme = scheme
        ctx.other_args = input.shape
        # ctx.saved = (x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, weight,
        #              running_mean, running_var, save_mean, save_var, training, eps, reserve)
        # ctx.saved = (x_mask_dct, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)
        # ctx.saved = (input, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)
        ctx.saved = (x_recover, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None

        # x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, weight, \
        #     running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        # x_dct, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        # x_mask_dct, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved
        input, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        # input_shape = ctx.other_args
        # input = mask_idct_2d(x_lfc_dct, x_hfc1_dct_ave, x_hfc2_dct_ave, x_hfc3_dct_ave, input_shape)
        # input = mask_idct_2d(x_dct)
        # input = decoder(x_mask_dct)

        empty_cache(config.empty_cache_threshold)

        if config.debug_memory_op_backward:
            global bn_layer_ct
            print("========== bn backward %d ==========" % bn_layer_ct)
            get_memory_usage(True)
            bn_layer_ct += 1

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None, \
               None, None


