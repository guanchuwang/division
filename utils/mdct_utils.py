
# import numpy as np
import torch
from utils.heap_utils import heap_create, heap_push, heap_pop
import torch_1_8_0_dct as torch_dct
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

# def auto_window(x, window_size, pixel=0.1):
#     batch_size, channel_num, step_num = x.shape
#     window_index = torch.zeros((batch_size, channel_num), dtype=int).cuda()
#
#     h = []
#     for batch_index in range(batch_size):
#         for channel_index in range(channel_num):
#             h.append((x[batch_index, channel_index, 0], batch_index, channel_index))
#     heap_create(h)
#
#     capacity = int(window_size * batch_size * channel_num / pixel)
#     barrier_num = int(1 / pixel)
#
#     for index in range(capacity):
#         x_min, batch_index, channel_index = heap_pop(h)
#         w = window_index[batch_index, channel_index]
#         w += 1
#         window_index[batch_index, channel_index] = w
#         if w < barrier_num:
#             heap_push(h, (x[batch_index, channel_index, w], batch_index, channel_index))
#
#     batch_window_size = window_index * pixel
#     return batch_window_size, window_index


def fd_feature(x_dct, barrier_num=8):
    batch_size, channel, height, width = x_dct.shape

    ####### has bug if height < barrier_num or width < barrier_num
    if height < barrier_num or width < barrier_num:
        pad2d = nn.ZeroPad2d(padding=(0, barrier_num-width, 0, barrier_num-height))
        x_dct = pad2d(x_dct)

    barrier_height = int(height/barrier_num) if height > barrier_num else 1
    barrier_width = int(width/barrier_num) if width > barrier_num else 1
    total_power = torch.square(x_dct).sum(dim=(2,3))

    # print(x_dct[0,0])
    # print(barrier_height, barrier_width, total_power[0,0])

    x_dct_block = torch.zeros((batch_size, channel, barrier_num, barrier_num)).cuda()
    for block_index1 in range(barrier_num):
        for block_index2 in range(barrier_num):
            index1_stt = block_index1 * barrier_height
            index1_end = block_index1 * barrier_height + barrier_height
            index2_stt = block_index2 * barrier_width
            index2_end = block_index2 * barrier_width + barrier_width
            # print(index1_stt, index1_end, index2_stt, index2_end)
            x_dct_block[:, :, block_index1, block_index2] = torch.square(x_dct[:, :, index1_stt:index1_end, index2_stt:index2_end]).sum(dim=(2,3))

    x_fd_feature_1 = torch.zeros((batch_size, channel, barrier_num)).cuda()
    # x_fd_feature_2 = torch.zeros((batch_size, channel, barrier_num)).cuda()
    # x_fd_feature_3 = torch.zeros((batch_size, channel, barrier_num)).cuda()
    for index in range(barrier_num):
        x_fd_feature_1[:, :, index] = x_dct_block[:, :, 0:index+1, 0:index+1].sum(dim=(2,3))/total_power
        # x_fd_feature_2[:, :, index] = x_dct_block[:, :, 0:index+1, :].sum(dim=(2,3))/total_power
        # x_fd_feature_3[:, :, index] = x_dct_block[:, :, :, 0:index+1].sum(dim=(2,3))/total_power

    # print("==============================")
    # print(x_dct[0,0])
    # print(x_dct_block[0,0])
    # print(x_fd_feature_1[0,0])
    # if channel == 512:
    #     hegsns

    return x_fd_feature_1 # x_fd_feature_2, x_fd_feature_3


def search_fdm_thd(x_fd_feature, window_size=1., max_search_time=8, min_window_size=0):

    thd_min, thd_max = 0., 1.5
    batch_size, channel, feature_num = x_fd_feature.shape
    # batch_size, channel, height, width = input_shape
    # barrier_num = height if height < feature_num else feature_num

    # remain_window_size = window_size
    remain_window_size = window_size - min_window_size/feature_num

    in_window_mask_1d = torch.zeros((batch_size, channel, feature_num-min_window_size)).cuda()
    for time_index in range(max_search_time):
        thd_prob = (thd_min + thd_max)/2
        in_window_mask_1d = (x_fd_feature[:, :, min_window_size:] < thd_prob)
        in_window_rate = (in_window_mask_1d.sum(dim=2)/feature_num).mean()
        if in_window_rate > remain_window_size:
            thd_max = thd_prob
        else:
            thd_min = thd_prob

    # in_window_mask_1d_comp = torch.cat((torch.ones((batch_size, channel, min_window_size)).type(torch.bool).cuda(),
    #                                     in_window_mask_1d), dim=2)

    # Channel Auto
    stair_value = torch.arange(in_window_mask_1d.shape[2]).cuda().unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size, channel, 1) + 0.5
    in_window_mask_1d_ave = (in_window_mask_1d.sum(dim=2).type(torch.float).mean(dim=0)).unsqueeze(dim=0).unsqueeze(dim=2).repeat((batch_size, 1, 1))
    in_window_mask_1d_valid = stair_value < in_window_mask_1d_ave
    in_window_mask_1d_comp = torch.cat((torch.ones((batch_size, channel, min_window_size)).type(torch.bool).cuda(),
                                        in_window_mask_1d_valid), dim=2)

    # stair_value = torch.arange(in_window_mask_1d.shape[2]).cuda().unsqueeze(dim=0).unsqueeze(dim=0).repeat(batch_size, channel, 1) + 0.5
    # in_window_mask_1d_ave = (in_window_mask_1d.sum(dim=2).type(torch.float).mean(dim=1)).unsqueeze(dim=1).unsqueeze(dim=2)
    # in_window_mask_1d_ = stair_value < in_window_mask_1d_ave
    # in_window_mask_1d_comp = torch.cat((torch.ones((batch_size, channel, min_window_size)).type(torch.bool).cuda(),
    #                                     in_window_mask_1d_), dim=2)

    # print(in_window_mask_1d_.shape)
    # print(in_window_mask_1d_[:, 0, :].sum(dim=1)/in_window_mask_1d_.shape[2])


    # print("thd_prob: {}".format(thd_prob))
    # print(window_size, (in_window_mask_1d_comp.sum(dim=2)/feature_num)[0])
    # print("===========================")
    # print("Channel num {}".format(x_fd_feature.shape[1]))
    # print(window_size, (in_window_mask_1d_comp.sum(dim=2)/feature_num).mean())
    # print(window_size, (in_window_mask_1d_ave/feature_num).mean())
    # print(window_size, (in_window_mask_1d_valid.sum(dim=2)/feature_num).mean())
    # print(window_size, in_window_mask_1d_comp[0,0])

    return in_window_mask_1d_comp

def generate_mask(in_window_mask_1d, input_shape):

    batch_size, channel, height, width = input_shape
    _, _, barrier_num = in_window_mask_1d.shape

    height = barrier_num if height < barrier_num else height
    width = barrier_num if width < barrier_num else width

    barrier_height = int(height / barrier_num)
    barrier_width = int(width / barrier_num)
    height_valid = barrier_height * barrier_num
    width_valid = barrier_width * barrier_num
    height_comp = height - height_valid
    width_comp = width - width_valid
    in_window_mask1_2d  = in_window_mask_1d.unsqueeze(dim=3).unsqueeze(dim=4).repeat(1,1,1, barrier_height, width).reshape(batch_size, channel, height_valid, width)
    in_window_mask2_2d  = in_window_mask_1d.unsqueeze(dim=3).unsqueeze(dim=4).repeat(1,1,1, barrier_width, height).reshape(batch_size, channel, width_valid, height).permute(0,1,3,2)

    # print(in_window_mask1_2d.shape, in_window_mask2_2d.shape, height_comp, width_comp)
    if height_comp > 0 or width_comp > 0:
        pad2d_height = nn.ZeroPad2d(padding=(0, 0, 0, height_comp))
        pad2d_width = nn.ZeroPad2d(padding=(0, width_comp, 0, 0))
        in_window_mask1_2d = pad2d_height(in_window_mask1_2d)
        in_window_mask2_2d = pad2d_width(in_window_mask2_2d)

    in_window_mask3_2d  =  in_window_mask1_2d &  in_window_mask2_2d
    out_window_mask1_2d = ~in_window_mask3_2d &  in_window_mask1_2d
    out_window_mask2_2d = ~in_window_mask3_2d &  in_window_mask2_2d
    out_window_mask3_2d = ~in_window_mask1_2d & ~in_window_mask2_2d & ~in_window_mask3_2d

    # print(in_window_mask1, in_window_mask2)
    # print(in_window_mask1.shape, in_window_mask2.shape)
    # hegsns


    return in_window_mask3_2d, out_window_mask1_2d, out_window_mask2_2d, out_window_mask3_2d


def mask(x_dct, in_window_mask_2d, out_window_mask1_2d, out_window_mask2_2d, out_window_mask3_2d):

    # print(x_dct.shape, in_window_mask1.shape, out_window_mask1.shape, out_window_mask2.shape, out_window_mask3.shape)
    # print(in_window_mask[0,0], out_window_mask1[0,0], out_window_mask2[0,0], out_window_mask3[0,0])

    batch_size, channel, height, width = x_dct.shape

    # print(in_window_mask_2d[0,0]  .type(torch.int))
    # print(out_window_mask1_2d[0,0].type(torch.int))
    # print(out_window_mask2_2d[0,0].type(torch.int))
    # print(out_window_mask3_2d[0,0].type(torch.int))

    in_window_mask_2d   = in_window_mask_2d[:, :, 0:height, 0:width]
    out_window_mask1_2d = out_window_mask1_2d[:, :, 0:height, 0:width]
    out_window_mask2_2d = out_window_mask2_2d[:, :, 0:height, 0:width]
    out_window_mask3_2d = out_window_mask3_2d[:, :, 0:height, 0:width]

    x_dct_encoded_in_window = x_dct * in_window_mask_2d
    ave_comp1 = (x_dct * out_window_mask1_2d).sum(dim=(2,3)) / (out_window_mask1_2d.sum(dim=(2,3)) + 1e-10)
    x_dct_encoded_out_window1 = ave_comp1.unsqueeze(dim=2).unsqueeze(dim=3) * out_window_mask1_2d
    ave_comp2 = (x_dct * out_window_mask2_2d).sum(dim=(2,3)) / (out_window_mask2_2d.sum(dim=(2,3)) + 1e-10)
    x_dct_encoded_out_window2 = ave_comp2.unsqueeze(dim=2).unsqueeze(dim=3) * out_window_mask2_2d
    ave_comp3 = (x_dct * out_window_mask3_2d).sum(dim=(2,3)) / (out_window_mask3_2d.sum(dim=(2,3)) + 1e-10)
    x_dct_encoded_out_window3 = ave_comp3.unsqueeze(dim=2).unsqueeze(dim=3) * out_window_mask3_2d

    # print(x_dct[0,0])
    # print(x_dct_encoded_in_window[0, 0])
    # print(in_window_mask_2d[0,0])
    # print(torch.abs(x_dct_encoded_in_window[0, 0]))
    # print(torch.abs(x_dct_encoded_in_window[0,0]).sum())
    # print((torch.abs(x_dct_encoded_in_window[0,0]).sum() < 1e-8))
    #
    # if torch.abs(x_dct_encoded_in_window[0,0]).sum() < 1e-8:
    #     hegsns

    # print(x_dct_encoded_in_window[0,0])
    # print(x_dct_encoded_out_window1[0,0])
    # print(x_dct_encoded_out_window2[0,0])
    # print(x_dct_encoded_out_window3[0,0])
    # print(x_dct_encoded_in_window[0,0] + x_dct_encoded_out_window1[0,0] + x_dct_encoded_out_window2[0,0] + x_dct_encoded_out_window3[0,0])
    # m1 = in_window_mask_2d[0,0].sum(dim=0)[0].type(torch.int)
    # m2 = in_window_mask_2d[0,0].sum(dim=1)[0].type(torch.int)
    # print(m1, m2)
    # print(x_dct[0,0,m1:,0:m2].mean())
    # print(x_dct[0,0,0:m1,m2:].mean())
    # print(x_dct[0,0,m1:,m2:].mean())
    # if channel == 512:
    #     hegsns

    return x_dct_encoded_in_window + x_dct_encoded_out_window1 + x_dct_encoded_out_window2 + x_dct_encoded_out_window3


def encoder(x, window_size, barrier_num, max_search_time, min_window_size):

    x_dct = torch_dct.dct_2d(x, norm='ortho')
    x_fd_feature = fd_feature(x_dct, barrier_num)
    in_window_mask_1d = search_fdm_thd(x_fd_feature, window_size, max_search_time, min_window_size)
    in_window_mask_2d, out_window_mask1_2d, out_window_mask2_2d, out_window_mask3_2d = generate_mask(in_window_mask_1d, x.shape)
    x_mask_dct = mask(x_dct, in_window_mask_2d, out_window_mask1_2d, out_window_mask2_2d, out_window_mask3_2d)

    return x_mask_dct


def decoder(x_mask_dct):

    return torch_dct.idct_2d(x_mask_dct)

# def encode(x_dct, batch_window_size):
#
#     batch_size, channel_num, height, width = x_dct.shape
#
#     m1 = torch.round(batch_window_size * x_dct.shape[2] + 0.5).type(torch.int).cuda()
#     m2 = torch.round(batch_window_size * x_dct.shape[3] + 0.5).type(torch.int).cuda()
#
#     x_lfc_dct = torch.zeros((m1 * m2).sum()).cuda()
#     x_hfc_ave1 = torch.zeros((batch_size, channel_num)).cuda()
#     x_hfc_ave2 = torch.zeros((batch_size, channel_num)).cuda()
#     x_hfc_ave3 = torch.zeros((batch_size, channel_num)).cuda()
#     insert_index = 0
#     for batch_index in range(batch_size):
#         for channel_index in range(channel_num):
#             m1_v = m1[batch_index, channel_index] # round(win_size * height + 0.5)
#             m2_v = m2[batch_index, channel_index] # round(win_size * width + 0.5)
#             x_lfc_dct[insert_index:insert_index + m1_v*m2_v] = x_dct[batch_index, channel_index, 0:m1_v, 0:m2_v].reshape(-1).clone()
#             x_hfc_ave1[batch_index, channel_index] = x_dct[batch_index, channel_index, m1_v:, 0:m2_v].mean().clone()
#             x_hfc_ave2[batch_index, channel_index] = x_dct[batch_index, channel_index, 0:m1_v, m2_v:].mean().clone()
#             x_hfc_ave3[batch_index, channel_index] = x_dct[batch_index, channel_index, m1_v:, m2_v:].mean().clone()
#             insert_index += m1_v*m2_v
#
#     return x_lfc_dct, x_hfc_ave1, x_hfc_ave2, x_hfc_ave3, m1, m2


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
