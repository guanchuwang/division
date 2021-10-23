import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import autocast
import cpp_extension.quantization as ext_quantization
import cpp_extension.minimax as ext_minimax
# import cpp_extension.backward_func as ext_backward_func
from torch.cuda.amp import autocast as autocast

import numpy as np
from conf import config
import time

class FDMP(Function):

    # @autocast()
    @staticmethod
    @torch.no_grad()
    def no_scheme_compute_quantization_bits(input, device):
        N = input.shape[0]
        D = input.shape[1]
        input_flatten = input.view(N, -1)
        num_features = input_flatten.shape[1]
        num_pixels = num_features // D

        # Compute min, max by groups
        if num_features % config.group_size != 0:
            # Padding
            new_num_features = (num_features // config.group_size + 1) * config.group_size
            delta = new_num_features - num_features
            input_flatten = torch.cat([input_flatten,
                                       torch.zeros([N, delta], dtype=input.dtype, device=input.device)], 1)

        input_groups = input_flatten.view(-1, config.group_size)
        mn, mx = ext_minimax.minimax(input_groups)

        b = config.hfc_bit_num
        return input_groups.view(N, -1, config.group_size), b, mn.view(N, -1, 1), mx.view(N, -1, 1)

    @staticmethod
    @torch.no_grad()
    def quantize_and_pack(data, bits, mn, mx):

        # Pack to bitstream
        if isinstance(bits, int):
            pack_func = ext_quantization.pack_single_precision
        else:
            pack_func = ext_quantization.pack_mixed_precision
        output, scale = pack_func(data, mn, mx, bits, True)

        return output, scale

    @staticmethod
    @torch.no_grad()
    def dequantize_and_unpack(data, shape, bits, scale, mn):

        # Pad to group_size
        N = shape[0]
        num_features = int(shape[1:].numel())
        group_size = config.group_size
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        # Unpack bitstream
        if isinstance(bits, int):
            unpack_func = ext_quantization.unpack_single_precision
        else:
            unpack_func = ext_quantization.unpack_mixed_precision

        data = unpack_func(data, bits, scale, mn, N, num_features // group_size, group_size)

        return data

    @staticmethod
    @torch.no_grad()
    @autocast()
    def fdmp(x, window_size, device=None):

        if window_size >= 1:
            return x, None, None, None, None

        if config.simulate:
            return FDMP.fdmp_simulation(x, window_size, device)

        x_lfc_dct, x_lfc, x_hfc = MDCT_op.freq_divide(x, window_size, device)
        # q_bits = int(quant_bit)

        # t0 = time.time()
        x_hfc_groups, q_bits, q_min, mx = FDMP.no_scheme_compute_quantization_bits(x_hfc, device)
        q_input, q_scale = FDMP.quantize_and_pack(x_hfc_groups, q_bits, q_min, mx)
        # torch.cuda.synchronize()

        if x.dtype == torch.float32:
            return x_lfc_dct, q_input, q_bits, q_scale.to(torch.bfloat16), q_min.to(torch.bfloat16)
        else:
            return x_lfc_dct, q_input, q_bits, q_scale, q_min

    @staticmethod
    @torch.no_grad()
    @autocast()
    def de_fdmp(feature_pack, q_input_shape, window_size, device=None):

        if window_size >= 1:
            x, _, _, _, _ = feature_pack
            return x

        if config.simulate:
            return FDMP.de_fdmp_simulation(feature_pack, q_input_shape, window_size, device)

        x_lfc_dct, q_input, q_bits, q_scale, q_min = feature_pack

        if q_scale.dtype == torch.bfloat16:
            q_scale = q_scale.to(torch.float32)
            q_min = q_min.to(torch.float32)
        x_hfc_dequant = FDMP.dequantize_and_unpack(q_input, q_input_shape, q_bits, q_scale, q_min)

        # Remove padding
        N = q_input_shape[0]
        num_features = int(q_input_shape[1:].numel())
        x_hfc_dequant = x_hfc_dequant.view(N, -1)[:, :num_features]
        x_hfc_dequant = x_hfc_dequant.view(*q_input_shape).contiguous()

        if window_size <= 0:
            return x_hfc_dequant

        # print(x_lfc_dct.shape)
        N = q_input_shape[-1]
        n = abs_window_size(N, window_size)
        dct_matrix = MDCT_op.generate_dct_matrix(N, n, device)
        x_lfc = MDCT_op.idct_2d(dct_matrix, x_lfc_dct)
        x = x_lfc + x_hfc_dequant

        return x
        # return x.to(torch.float)

    @staticmethod
    @torch.no_grad()
    def fdmp_simulation(x, window_size, device=None):

        x_lfc_dct, x_lfc, x_hfc = MDCT_op.freq_divide(x, window_size, device=None)
        x_min_group = x_hfc.min(dim=2)[0].min(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
        x_max_group = x_hfc.max(dim=2)[0].max(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=3)
        quant_step = (x_max_group - x_min_group) / (2 ** quant_bit) + 1e-8
        x_reference = x_min_group  # + quant_step/2
        x_hfc_quant = torch.round((x_hfc - x_reference) / quant_step)  # .type(torch.int)
        x_hfc_quant = torch.clamp(x_hfc_quant, min=0, max=2 ** quant_bit - 1).type(torch.int)

        return x_lfc_dct, x_hfc_quant, x_min_group, quant_step

    @staticmethod
    @torch.no_grad()
    def de_fdmp_simulation(feature_pack, q_input_shape, window_size, device=None):

        x_lfc_dct, x_hfc_quant, x_min_group, quant_step = feature_pack
        x_hfc_dequant = x_hfc_quant.type(torch.float) * quant_step + x_min_group

        if window_size <= 0:
            return x_hfc_dequant

        N = q_input_shape[-1]
        n = abs_window_size(N, window_size)
        dct_matrix = MDCT_op.generate_dct_matrix(N, n, device)

        x_lfc = MDCT_op.idct_2d(dct_matrix, x_lfc_dct)
        x = x_lfc + x_hfc_dequant

        return x


class MDCT_op(Function):

    @staticmethod
    @torch.no_grad()
    def generate_dct_matrix(N, n, device=None):

        i_vector = torch.arange(n, device=device)
        j_vector = torch.arange(N, device=device)

        i_matrix, j_matrix = torch.meshgrid(i_vector, j_vector)

        dct_matrix = torch.sqrt((1 + (i_matrix != 0) * 1) / N) * \
                     torch.cos((2 * j_matrix + 1) * 3.141592653589793 / (2 * N) * i_matrix)

        return dct_matrix
        # return torch.nn.Parameter(dct_matrix, requires_grad=False)

    @staticmethod
    @torch.no_grad()
    def dct_2d(dct_matrix, x):

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
        # return self.zero_padder(x)

    @staticmethod
    @torch.no_grad()
    def idct_2d(dct_matrix, x_dct):
        return MDCT_op.dct_2d(dct_matrix.T, x_dct)

    @staticmethod
    @torch.no_grad()
    def freq_divide(x, window_size, device=None):

        # print(x.device, self.dct_layer.weight.device)

        if window_size <= 0:
            return None, torch.zeros_like(x, device=device), x

        N = x.shape[-1]
        n = abs_window_size(N, window_size)
        dct_matrix = MDCT_op.generate_dct_matrix(N, n, device)
        x_lfc_dct = MDCT_op.dct_2d(dct_matrix, x)
        x_lfc = MDCT_op.idct_2d(dct_matrix, x_lfc_dct)

        return x_lfc_dct, x_lfc, x - x_lfc

    # def cuda(self, gpu):
    #
    #     self.dct_matrix.cuda(gpu)

@torch.no_grad()
def abs_window_size(N, window_size):
    if config.round_window:
        return round(window_size*N + 0.5)
    else:
        return round(window_size*N)

# class DCT_matrix:
#
#     def __init__(self):
#
#         self.dct_matrix_dict = nn.ParameterDict()
#
#     @torch.no_grad()
#     def new_dct_matrix(self, N, n):
#         print(len(self.dct_matrix_dict))
#
#         if N in self.dct_matrix_dict.keys():
#
#             return
#
#         self.dct_matrix_dict[str(N)] = MDCT_op.generate_dct_matrix(N, n)
#
#         return
#
#     def __getitem__(self, key):
#
#         return self.dct_matrix_dict[str(key)]
#
#
# dct_matrix_buf = DCT_matrix()



    # def dct(self, x):
    #
    #     # dct_matrix = self.generate_dct_matrix()
    #     # print("====================")
    #     # print(x.device, self.dct_layer.weight.device)
    #
    #     x_shape = x.shape
    #     x = x.view(-1, self.N)
    #     x = self.dct_layer(x)
    #     x = x.T.contiguous().view(-1, self.N)
    #     x = self.dct_layer(x).T
    #     x = x.view((self.n, self.n, x_shape[0], x_shape[1])).permute(2, 3, 0, 1)
    #
    #     return self.padder(x)
    #
    # def idct(self, x_dct):
    #
    #     return x_dct



# def generate_dct_matrix(N, n):
#
#     i_vector = torch.arange(n).cuda()
#     j_vector = torch.arange(N).cuda()
#
#     i_matrix, j_matrix = torch.meshgrid(i_vector, j_vector)
#
#     dct_matrix = torch.sqrt((1 + (i_matrix != 0) * 1) / N) \
#                  * torch.cos((2 * j_matrix + 1) * 3.14159265 / (2 * N) * i_matrix)
#
#     return dct_matrix
#
# def dct_2d(x, dct_matrix):
#
#     return torch.matmul(torch.matmul(dct_matrix, x), dct_matrix.T)
#
#
# def zero_padding(x, left, right, up, down):
#
#     padder = nn.ZeroPad2d((left, right, up, down))
#     return padder(x)

    # return F.pad(x, (left, right, up, down), "constant", 0)

