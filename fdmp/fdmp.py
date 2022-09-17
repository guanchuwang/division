import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import autocast
import cpp_extension.quantization as ext_quantization
# import exact.cpp_extension.quantization as ext_quantization
import cpp_extension.minimax as ext_minimax
# import cpp_extension.backward_func as ext_backward_func
# from torch.cuda.amp import autocast as autocast

# import numpy as np
from conf import config
import time

from utils import *

total_act_mem = 0
total_act_mem_lfc = 0 # torch.tensor(0).type(torch.long)
total_act_mem_hfc = 0 # torch.tensor(0).type(torch.long)

class FDMP_(Function):

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack(data, bits, mn, mx, N):

        # Pack to bitstream
        # print(pack_func)
        # print(bits)
        # scale = (2 ** bits - 1) / (mx - mn)

        mn_ = mn.view(N, 1, 1).repeat(1, data.shape[1], 1)
        mx_ = mx.view(N, 1, 1).repeat(1, data.shape[1], 1)

        # print(data.shape)
        # print(mn_.shape)
        # print(mx_.shape)
        # # print(scale.shape)
        # print(bits, type(bits))

        # output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
        output, scale = ext_quantization.pack_single_precision(data, mn_, mx_, bits, True)
        scale = scale[:,0,0].clone()
        # import pdb
        # pdb.set_trace()

        return output, scale

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack(data, shape, bits, scale, mn):

        # Pad to group_size
        Batch, Channel, Higth, Width = shape
        num_features = int(shape[2:].numel())

        if num_features > config.max_thread:
            mn_ = mn.view(Batch * Channel, 1, 1).repeat(1, Higth, 1)
            scale_ = scale.view(Batch * Channel, 1, 1).repeat(1, Higth, 1) # N, num_features // group_size, group_size)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, Higth, Width)

        else:
            mn_ = mn.view(Batch * Channel, 1, 1)
            scale_ = scale.view(Batch * Channel, 1, 1)
            data = ext_quantization.unpack_single_precision(data, bits, scale_, mn_, Batch * Channel, 1, Higth * Width)

        return data

    @staticmethod
    @torch.no_grad()
    def fdmp(x):

        Batch, Channel, Higth, Width = x.shape

        if Higth == 1:
            return x, None, None, None, None

        pool_kernel_size = config.lfc_block if Higth >= config.lfc_block else Higth
        x_lfc = F.avg_pool2d(x, pool_kernel_size, stride=pool_kernel_size, padding=0)
        x_lfc_float16 = x_lfc.to(torch.bfloat16)
        x_lfc_large = F.upsample_nearest(x_lfc_float16.to(x_lfc.dtype), size=(Higth, Width), scale_factor=None) # x_lfc.dtype

        # x_lfc_3d = x_lfc.reshape(Batch*Channel, x_lfc.shape[2], x_lfc.shape[3])
        # x_lfc_large_3da = F.interpolate(x_lfc_3d, size=(Width), mode='linear')
        # x_lfc_large_3db = F.interpolate(x_lfc_large_3da.permute(0,2,1), size=(Higth), mode='linear').permute(0,2,1)
        # x_lfc_large = x_lfc_large_3db.reshape(Batch, Channel, Higth, Width)
        # print(x.shape, x_lfc.shape, x_lfc_large.shape)

        x_hfc = x - x_lfc_large

        featuremap_area = Higth * Width # x_hfc.shape[-2:].numel()  # should be n

        if featuremap_area > config.max_thread:
            x_hfc_groups = x_hfc.reshape(Batch * Channel, Higth, Width)
        else:
            x_hfc_groups = x_hfc.reshape(Batch * Channel, 1, Higth * Width)

        q_min = x_hfc_groups.min(dim=-1).values.min(dim=-1).values
        mx = x_hfc_groups.max(dim=-1).values.max(dim=-1).values
        q_bits = config.hfc_bit_num
        q_input, q_scale = FDMP.quantize_and_pack(x_hfc_groups, q_bits, q_min, mx, Batch * Channel)

        return x_lfc_float16, q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)

        # return x_lfc, q_input, q_scale, q_min

        # if x.dtype == torch.float32:
        #     return x_lfc_float16, q_input, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16) # Remove x_lfc.to(torch.bfloat16) if accuracy drops
        # else:
        #     return x_lfc_float16, q_input, q_scale, q_min

    @staticmethod
    @torch.no_grad()
    def de_fdmp(feature_pack, q_input_shape):

        # if window_size >= 1:
        #     x, _, _, _, _ = feature_pack
        #     return x

        Batch, Channel, Higth, Width = q_input_shape

        if Higth == 1:
            x, _, _, _, _ = feature_pack
            return x

        x_lfc_float16, q_input, q_scale, q_min = feature_pack

        # Estimate valid group size
        if not config.half_precision:
            x_lfc = x_lfc_float16.to(torch.float32)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float32)
            q_min = q_min.to(torch.float32)
        else:
            x_lfc = x_lfc_float16.to(torch.float16)  # Remove it if accuracy drops
            q_scale = q_scale.to(torch.float16)
            q_min = q_min.to(torch.float16)

        q_bits = config.hfc_bit_num

        x_hfc_dequant = FDMP.dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)

        # Remove padding
        # num_features = q_input_shape[1:].numel()
        # x_hfc_dequant = x_hfc_dequant.view(q_input_shape[0], -1)[:, :num_features]
        x_hfc_dequant = x_hfc_dequant.view(*q_input_shape).contiguous()

        # pool_kernel_size = config.lfc_block if H >= config.lfc_block else H
        # x_lfc_large = F.interpolate(x_lfc, scale_factor=pool_kernel_size, mode='nearest')
        x_lfc_large = F.upsample_nearest(x_lfc, size=(Higth, Width), scale_factor=None)

        return x_lfc_large + x_hfc_dequant


    @staticmethod
    @torch.no_grad()
    def fdmp_simulation(x):

        Batch, Channel, Higth, Width = x.shape

        if Higth == 1:
            return x, None, None, None, None

        pool_kernel_size = config.lfc_block if Higth >= config.lfc_block else Higth
        x_lfc = F.avg_pool2d(x, pool_kernel_size, stride=pool_kernel_size, padding=0)
        x_lfc_large = F.upsample_nearest(x_lfc, size=(Higth, Width), scale_factor=None)
        x_hfc = x - x_lfc_large

        x_min_group = x_hfc.min(dim=2)[0].min(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
        x_max_group = x_hfc.max(dim=2)[0].max(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
        quant_step = (x_max_group - x_min_group) / (2 ** config.hfc_bit_num - 1) # + 1e-8
        x_reference = x_min_group  # + quant_step/2
        x_hfc_quant = torch.round((x_hfc - x_reference) / quant_step) # .type(torch.int)
        x_hfc_quant = torch.clamp(x_hfc_quant, min=0, max=2 ** config.hfc_bit_num - 1).type(torch.int)

        return x_lfc, x_hfc_quant, x_min_group, quant_step

    @staticmethod
    @torch.no_grad()
    def de_fdmp_simulation(feature_pack, q_input_shape):

        Batch, Channel, Higth, Width = q_input_shape
        if Higth == 1:
            x, _, _, _, _ = feature_pack
            return x, None, None, None, None

        x_lfc, x_hfc_quant, x_min_group, quant_step = feature_pack
        x_hfc_dequant = x_hfc_quant.type(torch.float) * quant_step + x_min_group

        x_lfc_large = F.upsample_nearest(x_lfc, size=(Higth, Width), scale_factor=None)
        x = x_lfc_large + x_hfc_dequant

        return x


class WDCT(Function):

    @staticmethod
    @torch.no_grad()
    def generate_dct_matrix(N, n, device):

        i_vector = torch.arange(n, device=device)
        j_vector = torch.arange(N, device=device)

        i_matrix, j_matrix = torch.meshgrid(i_vector, j_vector)

        dct_matrix = torch.sqrt((1 + (i_matrix != 0) * 1) / N) * \
                     torch.cos((2 * j_matrix + 1) * 3.141592653589793 / (2 * N) * i_matrix)

        return dct_matrix
        # return torch.nn.Parameter(dct_matrix, requires_grad=False)

    @staticmethod
    @torch.no_grad()
    def dct_1d_(dct_matrix, x):  # needs to debug

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        # x = layer(x)
        x = x.mm(dct_matrix.T)

        # print(x.shape)
        x = x.view((n2, x_shape[0])).permute(1, 0)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_1d_(dct_matrix, x_dct):
        return WDCT.dct_1d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_1d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_1d_(dct_matrix, x)

        else:
            return WDCT.idct_1d_(dct_matrix, x)

    @staticmethod
    @torch.no_grad()
    def dct_2d_(dct_matrix, x):
        # layer = self.idct_layer if inverse else self.dct_layer
        # n1, n2 = (self.n, self.N) if inverse else (self.N, self.n)
        # print("====================")
        # print(x.device, dct_matrix.device)

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        # x = layer(x)
        x = x.mm(dct_matrix.T)
        x = x.T.contiguous().view(-1, n1)
        # x = layer(x).T
        x = x.mm(dct_matrix.T).T

        x = x.view((n2, n2, x_shape[0], x_shape[1])).permute(2, 3, 0, 1)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_2d_(dct_matrix, x_dct):
        return WDCT.dct_2d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_2d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_2d_(dct_matrix, x)

        else:
            return WDCT.idct_2d_(dct_matrix, x)

    @staticmethod
    @torch.no_grad()
    def dct_3d_(dct_matrix, x):

        n2, n1 = dct_matrix.shape

        x_shape = x.shape
        x = x.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T)

        x = x.T.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T).T

        x = x.T.contiguous().view(-1, n1)
        x = x.mm(dct_matrix.T).T

        x = x.view((n2, n2, n2, x_shape[0], x_shape[1])).permute(2, 3, 4, 0, 1)

        return x

    @staticmethod
    @torch.no_grad()
    def idct_3d_(dct_matrix, x_dct):
        return WDCT.dct_3d_(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def dct_3d(dct_matrix, x, inverse=False):

        if not inverse:
            return WDCT.dct_3d_(dct_matrix, x)

        else:
            return WDCT.idct_3d_(dct_matrix, x)


class FDMP(Function):

    @staticmethod
    @torch.no_grad()
    def no_scheme_compute_quantization_bits(input, group_size):
        if not config.half_precision:
            return FDMP_.no_scheme_compute_quantization_bits(input, group_size)
        else:
            with autocast():
                return FDMP_.no_scheme_compute_quantization_bits(input, group_size)

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack(data, bits, mn, mx, N):
        if not config.half_precision:
            return FDMP_.quantize_and_pack(data, bits, mn, mx, N)
        else:
            with autocast():
                return FDMP_.quantize_and_pack(data, bits, mn, mx, N)

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack(data, shape, bits, scale, mn):
        if not config.half_precision:
            return FDMP_.dequantize_and_unpack(data, shape, bits, scale, mn)
        else:
            with autocast():
                return FDMP_.dequantize_and_unpack(data, shape, bits, scale, mn)

    @staticmethod
    @torch.no_grad()
    def freq_divide(x, dct_op, window_size, device):
        if not config.half_precision:
            return FDMP_.freq_divide(x, dct_op, window_size, device)
        else:
            with autocast():
                return FDMP_.freq_divide(x, dct_op, window_size, device)

    @staticmethod
    @torch.no_grad()
    def fdmp(x):
        if not config.half_precision:
            return FDMP_.fdmp(x)
        else:
            with autocast():
                return FDMP_.fdmp(x)

    @staticmethod
    @torch.no_grad()
    def de_fdmp(feature_pack, q_input_shape):
        if not config.half_precision:
            return FDMP_.de_fdmp(feature_pack, q_input_shape)
        else:
            with autocast():
                return FDMP_.de_fdmp(feature_pack, q_input_shape)

    @staticmethod
    @torch.no_grad()
    def fdmp_simulation(x):
        if not config.half_precision:
            return FDMP_.fdmp_simulation(x)
        else:
            with autocast():
                return FDMP_.fdmp_simulation(x)

    @staticmethod
    @torch.no_grad()
    def de_fdmp_simulation(feature_pack, q_input_shape):
        if not config.half_precision:
            return FDMP_.de_fdmp_simulation(feature_pack, q_input_shape)
        else:
            with autocast():
                return FDMP_.de_fdmp_simulation(feature_pack, q_input_shape)




