import re
import numpy as np



# extract_value('log.txt')

import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# import numpy as np
import json

with open("../../../debug/speed_debug_resnet50_division_b32.json") as json_file:
    speed_resnet_division_b32 = json.load(json_file)
with open("../../../debug/speed_debug_resnet50_division_b64.json") as json_file:
    speed_resnet_division_b64 = json.load(json_file)
with open("../../../debug/speed_debug_resnet50_division_b128.json") as json_file:
    speed_resnet_division_b128 = json.load(json_file)
with open("../../../debug/speed_debug_resnet50_division_b256.json") as json_file:
    speed_resnet_division_b256 = json.load(json_file)
with open("../../../debug/speed_debug_resnet50_division_b512.json") as json_file:
    speed_resnet_division_b512 = json.load(json_file)

 
# with open("../../../../actnn_exp/debug/speed_debug_resnet50_actnn_b32.json") as json_file:
#     speed_resnet_actnn_b32 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_resnet50_actnn_b64.json") as json_file:
    speed_resnet_actnn_b64 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_resnet50_actnn_b128.json") as json_file:
    speed_resnet_actnn_b128 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_resnet50_actnn_b256.json") as json_file:
    speed_resnet_actnn_b256 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_resnet50_actnn_b512.json") as json_file:
    speed_resnet_actnn_b512 = json.load(json_file)

with open("../../../../swap_imagenet/debug/speed_debug_resnet50_swap_b32.json") as json_file:
    speed_resnet_swap_b32 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_swap_b64.json") as json_file:
    speed_resnet_swap_b64 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_swap_b128.json") as json_file:
    speed_resnet_swap_b128 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_swap_b256.json") as json_file:
    speed_resnet_swap_b256 = json.load(json_file)
with open("../../../../swap_imagenet/debug/speed_debug_resnet50_swap_b512.json") as json_file:
    speed_resnet_swap_b512 = json.load(json_file)

with open("../../../../BLPA_imagenet/debug/speed_debug_resnet50_BLPA_b32.json") as json_file:
    speed_resnet_BLPA_b32 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_resnet50_BLPA_b64.json") as json_file:
    speed_resnet_BLPA_b64 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_resnet50_BLPA_b128.json") as json_file:
    speed_resnet_BLPA_b128 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_resnet50_BLPA_b256.json") as json_file:
    speed_resnet_BLPA_b256 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_resnet50_BLPA_b512.json") as json_file:
    speed_resnet_BLPA_b512 = json.load(json_file)

# with open("../../../../checkpoint_imagenet/debug/speed_debug_resnet50_checkpoint_b32.json") as json_file:
#     speed_resnet_checkpoint_b32 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_resnet50_checkpoint_b64.json") as json_file:
    speed_resnet_checkpoint_b64 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_resnet50_checkpoint_b128.json") as json_file:
    speed_resnet_checkpoint_b128 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_resnet50_checkpoint_b256.json") as json_file:
    speed_resnet_checkpoint_b256 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_resnet50_checkpoint_b512.json") as json_file:
    speed_resnet_checkpoint_b512 = json.load(json_file)

# with open("../../../../swap_imagenet/debug/speed_debug_resnet50_vanilla_b32.json") as json_file:
#     speed_resnet_vanilla_b32 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_vanilla_b64.json") as json_file:
    speed_resnet_vanilla_b64 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_vanilla_b128.json") as json_file:
    speed_resnet_vanilla_b128 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_resnet50_vanilla_b256.json") as json_file:
    speed_resnet_vanilla_b256 = json.load(json_file)
# with open("../../../../swap_imagenet/debug/speed_debug_resnet50_vanilla_b512.json") as json_file:
#     speed_resnet_vanilla_b512 = json.load(json_file)


with open("../../../debug/speed_debug_wrn50_2_division_b32.json") as json_file:
    speed_wrn_division_b32 = json.load(json_file)
with open("../../../debug/speed_debug_wrn50_2_division_b64.json") as json_file:
    speed_wrn_division_b64 = json.load(json_file)
with open("../../../debug/speed_debug_wrn50_2_division_b128.json") as json_file:
    speed_wrn_division_b128 = json.load(json_file)
with open("../../../debug/speed_debug_wrn50_2_division_b256.json") as json_file:
    speed_wrn_division_b256 = json.load(json_file)
with open("../../../debug/speed_debug_wrn50_2_division_b512.json") as json_file:
    speed_wrn_division_b512 = json.load(json_file)


# with open("../../../../actnn_exp/debug/speed_debug_wrn50_2_actnn_b32.json") as json_file:
#     speed_wrn_actnn_b32 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_wrn50_2_actnn_b64.json") as json_file:
    speed_wrn_actnn_b64 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_wrn50_2_actnn_b128.json") as json_file:
    speed_wrn_actnn_b128 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_wrn50_2_actnn_b256.json") as json_file:
    speed_wrn_actnn_b256 = json.load(json_file)
with open("../../../../actnn_exp/debug/speed_debug_wrn50_2_actnn_b512.json") as json_file:
    speed_wrn_actnn_b512 = json.load(json_file)

with open("../../../../swap_imagenet/debug/speed_debug_wrn50_2_swap_b32.json") as json_file:
    speed_wrn_swap_b32 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_swap_b64.json") as json_file:
    speed_wrn_swap_b64 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_swap_b128.json") as json_file:
    speed_wrn_swap_b128 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_swap_b256.json") as json_file:
    speed_wrn_swap_b256 = json.load(json_file)
with open("../../../../swap_imagenet/debug/speed_debug_wrn50_2_swap_b512.json") as json_file:
    speed_wrn_swap_b512 = json.load(json_file)
#
with open("../../../../BLPA_imagenet/debug/speed_debug_wrn50_2_BLPA_b32.json") as json_file:
    speed_wrn_BLPA_b32 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_wrn50_2_BLPA_b64.json") as json_file:
    speed_wrn_BLPA_b64 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_wrn50_2_BLPA_b128.json") as json_file:
    speed_wrn_BLPA_b128 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_wrn50_2_BLPA_b256.json") as json_file:
    speed_wrn_BLPA_b256 = json.load(json_file)
with open("../../../../BLPA_imagenet/debug/speed_debug_wrn50_2_BLPA_b512.json") as json_file:
    speed_wrn_BLPA_b512 = json.load(json_file)
#
# with open("../../../../checkpoint_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_checkpoint_b32.json") as json_file:
#     speed_wrn_checkpoint_b32 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_wrn50_2_checkpoint_b64.json") as json_file:
    speed_wrn_checkpoint_b64 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_wrn50_2_checkpoint_b128.json") as json_file:
    speed_wrn_checkpoint_b128 = json.load(json_file)
with open("../../../../checkpoint_imagenet/debug/speed_debug_wrn50_2_checkpoint_b256.json") as json_file:
    speed_wrn_checkpoint_b256 = json.load(json_file)
# with open("../../../../checkpoint_imagenet/debug/speed_debug_wrn50_2_checkpoint_b512.json") as json_file:
#     speed_wrn_checkpoint_b512 = json.load(json_file)
#
# with open("../../../../swap_imagenet/debug/speed_debug_wrn50_2_vanilla_b32.json") as json_file:
#     speed_wrn_vanilla_b32 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_vanilla_b64.json") as json_file:
    speed_wrn_vanilla_b64 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_vanilla_b128.json") as json_file:
    speed_wrn_vanilla_b128 = json.load(json_file)
with open("../../../../swap_imagenet/debug/iclr23_rebuttal/speed_debug_wrn50_2_vanilla_b256.json") as json_file:
    speed_wrn_vanilla_b256 = json.load(json_file)
# with open("../../../../swap_imagenet/debug/speed_debug_wrn50_2_vanilla_b512.json") as json_file:
#     speed_wrn_vanilla_b512 = json.load(json_file)



# with open("../debug/speed_debug_densenet161_division_b32.json") as json_file:
#     speed_densenet_division_b32 = json.load(json_file)
# with open("../debug/speed_debug_densenet161_division_b64.json") as json_file:
#     speed_densenet_division_b64 = json.load(json_file)
# with open("../debug/speed_debug_densenet161_division_b128.json") as json_file:
#     speed_densenet_division_b128 = json.load(json_file)
# with open("../debug/speed_debug_densenet161_division_b256.json") as json_file:
#     speed_densenet_division_b256 = json.load(json_file)
# with open("../debug/speed_debug_densenet161_division_b512.json") as json_file:
#     speed_densenet_division_b512 = json.load(json_file)

# with open("../debug/actnn_debug/speed_debug_densenet161_actnn_b32.json") as json_file:
#     speed_densenet_actnn_b32 = json.load(json_file)
# with open("../debug/actnn_debug/speed_debug_densenet161_actnn_b64.json") as json_file:
#     speed_densenet_actnn_b64 = json.load(json_file)
# with open("../debug/actnn_debug/speed_debug_densenet161_actnn_b128.json") as json_file:
#     speed_densenet_actnn_b128 = json.load(json_file)
# with open("../debug/actnn_debug/speed_debug_densenet161_actnn_b256.json") as json_file:
#     speed_densenet_actnn_b256 = json.load(json_file)
# with open("../debug/actnn_debug/speed_debug_densenet161_actnn_b512.json") as json_file:
#     speed_densenet_actnn_b512 = json.load(json_file)



# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_swap_b32.json") as json_file:
#     speed_densenet_swap_b32 = json.load(json_file)# speed_densenet_division_b32  = json.loads("debug/speed_debug_densenet161_division_b32.json")
# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_swap_b64.json") as json_file:# speed_densenet_division_b64  = json.loads("debug/speed_debug_densenet161_division_b64.json")
#     speed_densenet_swap_b64 = json.load(json_file)# speed_densenet_division_b128 = json.loads("debug/speed_debug_densenet161_division_b128.json")
# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_swap_b128.json") as json_file:# speed_densenet_division_b256 = json.loads("debug/speed_debug_densenet161_division_b256.json")
#     speed_densenet_swap_b128 = json.load(json_file)# speed_densenet_division_b512 = json.loads("debug/speed_debug_densenet161_division_b512.json")
# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_swap_b256.json") as json_file:
#     speed_densenet_swap_b256 = json.load(json_file)
# # with open("../../../../swap_imagenet/debug/speed_debug_densenet161_swap_b512.json") as json_file:
# #     speed_densenet_swap_b512 = json.load(json_file)

# with open("../../../../BLPA_imagenet/debug/speed_debug_densenet161_BLPA_b32.json") as json_file:
#     speed_densenet_BLPA_b32 = json.load(json_file)# speed_densenet_division_b32  = json.loads("debug/speed_debug_densenet161_division_b32.json")
# with open("../../../../BLPA_imagenet/debug/speed_debug_densenet161_BLPA_b64.json") as json_file:# speed_densenet_division_b64  = json.loads("debug/speed_debug_densenet161_division_b64.json")
#     speed_densenet_BLPA_b64 = json.load(json_file)# speed_densenet_division_b128 = json.loads("debug/speed_debug_densenet161_division_b128.json")
# with open("../../../../BLPA_imagenet/debug/speed_debug_densenet161_BLPA_b128.json") as json_file:# speed_densenet_division_b256 = json.loads("debug/speed_debug_densenet161_division_b256.json")
#     speed_densenet_BLPA_b128 = json.load(json_file)# speed_densenet_division_b512 = json.loads("debug/speed_debug_densenet161_division_b512.json")
# with open("../../../../BLPA_imagenet/debug/speed_debug_densenet161_BLPA_b256.json") as json_file:
#     speed_densenet_BLPA_b256 = json.load(json_file)
# with open("../../../../BLPA_imagenet/debug/speed_debug_densenet161_BLPA_b512.json") as json_file:
#     speed_densenet_BLPA_b512 = json.load(json_file)

# with open("../../../../checkpoint_imagenet/debug/speed_debug_densenet161_checkpoint_b32.json") as json_file:
#     speed_densenet_checkpoint_b32 = json.load(json_file)# speed_densenet_division_b32  = json.loads("debug/speed_debug_densenet161_division_b32.json")
# with open("../../../../checkpoint_imagenet/debug/speed_debug_densenet161_checkpoint_b64.json") as json_file:# speed_densenet_division_b64  = json.loads("debug/speed_debug_densenet161_division_b64.json")
#     speed_densenet_checkpoint_b64 = json.load(json_file)# speed_densenet_division_b128 = json.loads("debug/speed_debug_densenet161_division_b128.json")
# with open("../../../../checkpoint_imagenet/debug/speed_debug_densenet161_checkpoint_b128.json") as json_file:# speed_densenet_division_b256 = json.loads("debug/speed_debug_densenet161_division_b256.json")
#     speed_densenet_checkpoint_b128 = json.load(json_file)# speed_densenet_division_b512 = json.loads("debug/speed_debug_densenet161_division_b512.json")
# with open("../../../../checkpoint_imagenet/debug/speed_debug_densenet161_checkpoint_b256.json") as json_file:
#     speed_densenet_checkpoint_b256 = json.load(json_file)
# # with open("../../../../checkpoint_imagenet/debug/speed_debug_densenet161_checkpoint_b512.json") as json_file:
# #     speed_densenet_checkpoint_b512 = json.load(json_file)

# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_vanilla_b64.json") as json_file:
#     speed_densenet_vanilla_b64 = json.load(json_file)#
# with open("../../../../swap_imagenet/debug/speed_debug_densenet161_vanilla_b128.json") as json_file:
#     speed_densenet_vanilla_b128 = json.load(json_file)#



bar_width = 1  
bactch_size_buf = np.array([64, 128, 256, 512])
# epoch_cifar10 = 50


speed_resnet50_division = [ # speed_resnet_division_b32['ips'],
                           speed_resnet_division_b64['ips'],
                           speed_resnet_division_b128['ips'],
                           speed_resnet_division_b256['ips'],
                           speed_resnet_division_b512['ips']
                           ]

speed_resnet50_actnn = [ # speed_resnet_actnn_b32['ips'],
                        speed_resnet_actnn_b64['ips'],
                        speed_resnet_actnn_b128['ips'],
                        speed_resnet_actnn_b256['ips'],
                        speed_resnet_actnn_b512['ips']
                           ]

speed_resnet50_swap = [ # speed_resnet_swap_b32['ips'],
                       speed_resnet_swap_b64['ips'],
                       speed_resnet_swap_b128['ips'],
                       speed_resnet_swap_b256['ips'],
                       0, # speed_resnet_swap_b512['ips']
                           ]

speed_resnet50_BLPA = [ # speed_resnet_BLPA_b32['ips'],
                       speed_resnet_BLPA_b64['ips'],
                       speed_resnet_BLPA_b128['ips'],
                       speed_resnet_BLPA_b256['ips'],
                       speed_resnet_BLPA_b512['ips']
                           ]

speed_resnet50_checkpoint = [ # speed_resnet_checkpoint_b32['ips'],
                             speed_resnet_checkpoint_b64['ips'],
                             speed_resnet_checkpoint_b128['ips'],
                             speed_resnet_checkpoint_b256['ips'],
                             speed_resnet_checkpoint_b512['ips'],
                           ]

speed_resnet50_vanilla = [ #  speed_resnet_vanilla_b32['ips'],
                          speed_resnet_vanilla_b64['ips'],
                          speed_resnet_vanilla_b128['ips'],
                          speed_resnet_vanilla_b256['ips'],
                          0, # speed_resnet_vanilla_b512['ips']
                          ]

speed_wrn50_2_division = [ # speed_wrn_division_b32['ips'],
                           speed_wrn_division_b64['ips'],
                           speed_wrn_division_b128['ips'],
                           speed_wrn_division_b256['ips'],
                           speed_wrn_division_b512['ips']
                           ]

speed_wrn50_2_actnn = [ # speed_wrn_actnn_b32['ips'],
                        speed_wrn_actnn_b64['ips'],
                        speed_wrn_actnn_b128['ips'],
                        speed_wrn_actnn_b256['ips'],
                        speed_wrn_actnn_b512['ips']
                           ]

speed_wrn50_2_swap = [ #  speed_wrn_swap_b32['ips'],
                       speed_wrn_swap_b64['ips'],
                       speed_wrn_swap_b128['ips'],
                       speed_wrn_swap_b256['ips'],
                       0, # speed_wrn_swap_b512['ips']
                           ]
#
speed_wrn50_2_BLPA = [ #  speed_wrn_BLPA_b32['ips'],
                       speed_wrn_BLPA_b64['ips'],
                       speed_wrn_BLPA_b128['ips'],
                       speed_wrn_BLPA_b256['ips'],
                       speed_wrn_BLPA_b512['ips']
                           ]
#
speed_wrn50_2_checkpoint = [ #  speed_wrn_checkpoint_b32['ips'],
                             speed_wrn_checkpoint_b64['ips'],
                             speed_wrn_checkpoint_b128['ips'],
                             speed_wrn_checkpoint_b256['ips'],
                             0, # speed_wrn_checkpoint_b512['ips']
                           ]
#
speed_wrn50_2_vanilla = [ #  speed_wrn_vanilla_b32['ips'],
                          speed_wrn_vanilla_b64['ips'],
                          speed_wrn_vanilla_b128['ips'],
                          speed_wrn_vanilla_b256['ips'],
                          0, # speed_wrn_vanilla_b512['ips']
                          ]

# speed_densenet161_division = [# speed_densenet_division_b32['ips'],
#                            speed_densenet_division_b64['ips'],
#                            speed_densenet_division_b128['ips'],
#                            speed_densenet_division_b256['ips'],
#                            speed_densenet_division_b512['ips']
#                            ]
#
# speed_densenet161_actnn = [# speed_densenet_actnn_b32['ips'],
#                            speed_densenet_actnn_b64['ips'],
#                            speed_densenet_actnn_b128['ips'],
#                            speed_densenet_actnn_b256['ips'],
#                            speed_densenet_actnn_b512['ips']
#                            ]
#
# speed_densenet161_swap = [# speed_densenet_swap_b32['ips'],
#                           speed_densenet_swap_b64['ips'],
#                           speed_densenet_swap_b128['ips'],
#                           speed_densenet_swap_b256['ips'],
#                           # speed_densenet_swap_b512['ips']
#                           ]
#
# speed_densenet161_BLPA = [# speed_densenet_BLPA_b32['ips'],
#                           speed_densenet_BLPA_b64['ips'],
#                           speed_densenet_BLPA_b128['ips'],
#                           speed_densenet_BLPA_b256['ips'],
#                           speed_densenet_BLPA_b512['ips']
#                           ]
#
# speed_densenet161_checkpoint = [# speed_densenet_checkpoint_b32['ips'],
#                                 speed_densenet_checkpoint_b64['ips'],
#                                 speed_densenet_checkpoint_b128['ips'],
#                                 speed_densenet_checkpoint_b256['ips'],
#                                 # speed_densenet_checkpoint_b512['ips']
#                                 ]
#
# speed_densenet161_vanilla = [# speed_densenet_checkpoint_b32['ips'],
#                                 speed_densenet_vanilla_b64['ips'],
#                                 speed_densenet_vanilla_b128['ips'],
#                                 # speed_densenet_checkpoint_b256['ips'],
#                                 # speed_densenet_checkpoint_b512['ips']
#                                 ]



division_resnet50_axis = np.arange(len(speed_resnet50_division)) * 8 * bar_width + bar_width
vanilla_resnet50_axis = division_resnet50_axis - bar_width
# BLPA_resnet50_axis = actnn_resnet50_axis + bar_width
checkpoint_resnet50_axis = division_resnet50_axis + bar_width
actnn_resnet50_axis = checkpoint_resnet50_axis + bar_width
swap_resnet50_axis = actnn_resnet50_axis + bar_width

plt.bar(vanilla_resnet50_axis, speed_resnet50_vanilla, width=bar_width, color="b", label="Normal")
plt.bar(division_resnet50_axis, speed_resnet50_division, width=bar_width, color="r", label="DIVISION")
# plt.bar(BLPA_resnet50_axis, speed_resnet50_BLPA, width=bar_width, color="darkgoldenrod", label="BLPA")
plt.bar(checkpoint_resnet50_axis, speed_resnet50_checkpoint, width=bar_width, color="darkgoldenrod", label="Checkpoint")
plt.bar(actnn_resnet50_axis, speed_resnet50_actnn, width=bar_width, color="g", label="ActNN")
plt.bar(swap_resnet50_axis, speed_resnet50_swap, width=bar_width, color="darkmagenta", label="SWAP")

print(speed_resnet50_division)
# print(speed_resnet50_actnn)
# print(speed_resnet50_vanilla)
print(speed_resnet50_checkpoint)

plt.text(vanilla_resnet50_axis[-1], speed_resnet50_vanilla[-1], "x", size=25, color="b", fontweight="bold")
# plt.text(swap_resnet50_axis[-2], speed_resnet50_swap[-2], "x", size=25, color="darkmagenta", fontweight="bold")
plt.text(swap_resnet50_axis[-1], speed_resnet50_swap[-1], "x", size=25, color="darkmagenta", fontweight="bold")
# plt.text(checkpoint_resnet50_axis[-1], speed_resnet50_checkpoint[-1], "x", size=25, color="darkgoldenrod", fontweight="bold")

plt.xticks(division_resnet50_axis + bar_width, bactch_size_buf, fontsize=30)
plt.yticks(fontsize=30)
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# plt.gca().ticklabel_format(style='sci', scilimits=(0,2), axis='y')
plt.xlim([division_resnet50_axis[0] - 3*bar_width, division_resnet50_axis[-1] + 6*bar_width])
plt.ylim([0, 850])
plt.xlabel("Batch-size", fontsize=30)
plt.ylabel("Throughput", fontsize=30)
plt.legend(loc='upper left', fontsize=25, ncol=2)
plt.grid(axis='y')

plt.subplots_adjust(left=0.16, bottom=0.145, top=0.96, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)
plt.savefig("../figure/resnet50_throughput_imagenet.png")

plt.show()
# plt.close()


division_wrn50_2_axis = np.arange(len(speed_wrn50_2_division)) * 8 * bar_width + bar_width
vanilla_wrn50_2_axis = division_wrn50_2_axis - bar_width
# BLPA_wrn50_2_axis = actnn_wrn50_2_axis + bar_width
checkpoint_wrn50_2_axis = division_wrn50_2_axis + bar_width
actnn_wrn50_2_axis = checkpoint_wrn50_2_axis + bar_width
swap_wrn50_2_axis = actnn_wrn50_2_axis + bar_width

plt.bar(vanilla_wrn50_2_axis, speed_wrn50_2_vanilla, width=bar_width, color="b", label="Normal")
plt.bar(division_wrn50_2_axis, speed_wrn50_2_division, width=bar_width, color="r", label="DIVISION")
# plt.bar(BLPA_wrn50_2_axis, speed_wrn50_2_BLPA, width=bar_width, color="darkgoldenrod", label="BLPA")
plt.bar(checkpoint_wrn50_2_axis, speed_wrn50_2_checkpoint, width=bar_width, color="darkgoldenrod", label="Checkpoint")
plt.bar(actnn_wrn50_2_axis, speed_wrn50_2_actnn, width=bar_width, color="g", label="ActNN")
plt.bar(swap_wrn50_2_axis, speed_wrn50_2_swap, width=bar_width, color="darkmagenta", label="SWAP")

# print(speed_wrn50_2_vanilla)
# print(speed_wrn50_2_actnn)
# print(speed_wrn50_2_division)
# print(speed_wrn50_2_checkpoint)

plt.text(vanilla_wrn50_2_axis[-1], speed_wrn50_2_vanilla[-1], "x", size=25, color="b", fontweight="bold")
# plt.text(swap_wrn50_2_axis[-2], speed_wrn50_2_swap[-2], "x", size=25, color="darkmagenta", fontweight="bold")
plt.text(swap_wrn50_2_axis[-1], speed_wrn50_2_swap[-1], "x", size=25, color="darkmagenta", fontweight="bold")
plt.text(checkpoint_wrn50_2_axis[-1], speed_wrn50_2_checkpoint[-1], "x", size=25, color="darkgoldenrod", fontweight="bold")

plt.xticks(division_wrn50_2_axis + bar_width, bactch_size_buf, fontsize=30)
plt.yticks(fontsize=30)
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# plt.gca().ticklabel_format(style='sci', scilimits=(0,2), axis='y')
plt.xlim([division_wrn50_2_axis[0] - 3*bar_width, division_wrn50_2_axis[-1] + 6*bar_width])
plt.ylim([0, 550])
plt.xlabel("Batch-size", fontsize=30)
plt.ylabel("Throughput", fontsize=30)
plt.legend(loc='upper left', fontsize=25, ncol=2)
plt.grid(axis='y')

plt.subplots_adjust(left=0.16, bottom=0.145, top=0.96, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)
plt.savefig("../figure/wrn50_2_throughput_imagenet.png")

plt.show()
# plt.close()

                                     
# division_densenet161_axis = np.arange(len(speed_densenet161_division)) * 8 * bar_width + bar_width
# vanilla_densenet161_axis = division_densenet161_axis[:-2] - bar_width
# actnn_densenet161_axis = division_densenet161_axis + bar_width
# BLPA_densenet161_axis = actnn_densenet161_axis + bar_width
# checkpoint_densenet161_axis = actnn_densenet161_axis[:-1] + 2*bar_width
# swap_densenet161_axis = actnn_densenet161_axis[:-1] + 3*bar_width
#
# plt.bar(vanilla_densenet161_axis, speed_densenet161_vanilla, width=bar_width, color="b", label="Vanilla")
# plt.bar(division_densenet161_axis, speed_densenet161_division, width=bar_width, color="r", label="DIVISION")
# plt.bar(actnn_densenet161_axis, speed_densenet161_actnn, width=bar_width, color="g", label="ActNN")
# plt.bar(BLPA_densenet161_axis, speed_densenet161_BLPA, width=bar_width, color="darkgoldenrod", label="BLPA")
# plt.bar(checkpoint_densenet161_axis, speed_densenet161_checkpoint, width=bar_width, color="darkmagenta", label="Checkpoint")
# plt.bar(swap_densenet161_axis, speed_densenet161_swap, width=bar_width, color="pink", label="SWAP")
#
# # plt.text(x,
# # 	y,
# # 	string,
# # 	fontsize=15,
# # 	verticalalignment="top",
# # 	horizontalalignment="right"
# # )
#
# # plt.scatter([200], [200], marker='x', color="r", s=200)
# plt.xticks(division_densenet161_axis + bar_width, bactch_size_buf, fontsize=30)
# plt.yticks(fontsize=30)
# # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# # plt.gca().ticklabel_format(style='sci', scilimits=(0,2), axis='y')
# plt.xlim([division_densenet161_axis[0] - 2*bar_width, division_densenet161_axis[-1] + 6*bar_width])
# plt.ylim([0, 200])
# plt.xlabel("Batch-size", fontsize=30)
# plt.ylabel("Throughput", fontsize=30)
# plt.legend(loc='best', fontsize=18, ncol=2)
#
# plt.subplots_adjust(left=0.115, bottom=0.115, top=0.975, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)
# plt.savefig("../figure/densenet161_throughput_imagenet.pdf")
#
# plt.show()
# # plt.close()

