import re
import numpy as np

def extract_value(fname, regex_str, strloc, endloc):
    f = open(fname, 'r')
    str_buf = f.read()
    # print(str_buf, regex_str)
    regex = re.compile(regex_str)

    str_list = regex.findall(str_buf)
    # print(str_list)
    data_list = []
    for item in str_list:
        # data_list.append(float(item[7:]))
        x = item[strloc:endloc]
        # print(x)
        data_list.append(float(x))

    print(data_list)
    # print(len(data_list))
    return data_list

# extract_value('log.txt')

import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# import numpy as np


# # acc_baseline = extract_value('../log/cifar10/resnet18_lr_0.1_win_1.log', 'Acc@1 .+ Acc@5', 6, -6)
# # acc_baseline_b = extract_value('../log/cifar10/resnet18_lr_0.1_win_1_b.log', 'Acc@1 .+ Acc@5', 6, -6)
acc_baseline_c = extract_value('../log/cifar10/resnet18_cifar10_baseline.log', 'Acc@1 .+ Acc@5', 6, -6)
# # acc_baseline_adam = extract_value('../log/cifar10/dctb_resnet18_lr_0001_win_1_adam.log', 'Acc@1 .+ Acc@5', 6, -6)
fdq_resnet18_lr_01_lf_8_hq_2 = extract_value('../log/cifar10/resnet18_cifar10_lb_8_hq_2.log', 'Acc@1 .+ Acc@5', 6, -6)

# cifar100_resnet18_baseline_coslr_e_100_b_128  = extract_value('../log/cifar100/resnet18_cifar100_baseline_coslr_e_100.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet18_baseline_coslr_e_200_b_128  = extract_value('../log/cifar100/resnet18_cifar100_baseline_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet18_baseline_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet18_cifar100_baseline_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet18_baseline_steplr_e_100_b_128 = extract_value('../log/cifar100/resnet18_cifar100_baseline_steplr_e_100.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet18_lr_01_lf_8_hq_2_coslr_e_100_b_128  = extract_value('../log/cifar100/resnet18_cifar100_lb_8_hq_2_coslr_e_100.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet18_lr_01_lf_8_hq_2_coslr_e_200_b_128  = extract_value('../log/cifar100/resnet18_cifar100_lb_8_hq_2_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet18_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet18_cifar100_lb_8_hq_2_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet18_lr_01_lf_8_hq_2_steplr_e_100_b_128 = extract_value('../log/cifar100/resnet18_cifar100_lb_8_hq_2_steplr_e_100.log', 'Acc@1 .+ Acc@5', 6, -6)

# cifar100_resnet34_baseline_coslr_e_200_b_128 = extract_value('../log/cifar100/resnet34_cifar100_baseline_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet34_baseline_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet34_cifar100_baseline_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet34_lr_01_lf_8_hq_2_coslr_e_200_b_128 = extract_value('../log/cifar100/resnet34_cifar100_lb_8_hq_2_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet34_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet34_cifar100_lb_8_hq_2_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet34_baseline_coslr_e_200_b_256 = extract_value('../log/cifar100/resnet34_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_resnet34_baseline_steplr_e_200_b_256 = extract_value('../log/cifar100/resnet34_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet34_lr_01_lf_8_hq_2_coslr_e_200_b_256 = extract_value('../log/cifar100/resnet34_cifar100_lb_8_hq_2_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_resnet34_lr_01_lf_8_hq_2_steplr_e_200_b_256 = extract_value('../log/cifar100/resnet34_cifar100_lb_8_hq_2_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)

cifar100_resnet50_baseline_coslr_e_200_b_128 = extract_value('../log/cifar100/resnet50_cifar100_baseline_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet50_cifar100_baseline_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_128 = extract_value('../log/cifar100/resnet50_cifar100_lb_8_hq_2_coslr_e_200_b_128_s_7.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../log/cifar100/resnet50_cifar100_lb_8_hq_2_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_coslr_e_200_b_256 = extract_value('../log/cifar100/resnet50_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_steplr_e_200_b_256 = extract_value('../log/cifar100/resnet50_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_256 = extract_value('../log/cifar100/resnet50_cifar100_lb_8_hq_2_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_256 = extract_value('../log/cifar100/resnet50_cifar100_lb_8_hq_2_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)

# cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_128 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_coslr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_128 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_256 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_256 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_1024 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_coslr_e_200_b_1024.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_1024 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_baseline_steplr_e_200_b_1024.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_shufflenet_v2_x1_0_lr_01_lf_8_hq_2_coslr_e_200_b_1024 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_lb_8_hq_2_coslr_e_200_b_1024.log', 'Acc@1 .+ Acc@5', 6, -6)
# cifar100_fdq_shufflenet_v2_x1_0_lr_01_lf_8_hq_2_steplr_e_200_b_1024 = extract_value('../log/cifar100/shufflenet_v2_x1_0_cifar100_lb_8_hq_2_steplr_e_200_b_1024.log', 'Acc@1 .+ Acc@5', 6, -6)


cifar100_densenet121_baseline_coslr_e_200_b_128 = extract_value('../log/cifar100/densenet121_cifar100_baseline_coslr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_steplr_e_200_b_128 = extract_value('../log/cifar100/densenet121_cifar100_baseline_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_coslr_e_200_b_256 = extract_value('../log/cifar100/densenet121_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_steplr_e_200_b_256 = extract_value('../log/cifar100/densenet121_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../log/cifar100/densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_256 = extract_value('../log/cifar100/densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)


# # imagenet_mobilenet_baseline_sgd_lr_005_wd_00004_a = extract_value('../log/mobilenet/mobilenet_imagenet_baseline_lr_005_wd_00004_epoch60.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_baseline_sgd_lr_005_wd_00004_b = extract_value('../log/mobilenet/mobilenet_baseline_imagenet_lr_005_wd_00004.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_baseline_sgd_coslr_005_wd_00004 = extract_value('../log/mobilenet/mobilenet_imagenet_baseline_coslr_005_wd_00004_2.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_baseline_sgd_lr_01_wd_0001 = extract_value('../log/mobilenet/mobilenet_imagenet_baseline_lr_01_wd_0001.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_fdq_win_0_1_hfc_2_sgd = extract_value('../log/mobilenet/mobilenet_imagenet_fdq_win_0_1_hfc_2_lr_005_wd_00004.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_fdq_win_0_2_hfc_2_sgd = extract_value('../log/mobilenet/mobilenet_imagenet_fdq_win_0_2_hfc_2_lr_005_wd_00004.log', 'Acc@1 .+ Acc@5', 6, -6)
# imagenet_mobilenet_fdq_win_0_1_hfc_2_sgd_simulate = extract_value('../log/mobilenet/mobilenet_imagenet_fdq_win_0_1_hfc_2_lr_005_wd_00004_simulate.log', 'Acc@1 .+ Acc@5', 6, -6)

plot_list = [

    # # CIFAR 10
    {"data": acc_baseline_c, "label": "baseline SGD"},
    {"data": fdq_resnet18_lr_01_lf_8_hq_2, "label": "FDQ Resnet18 lb 8 hq 2"},

    # CIFAR 100 Resnet18
    # {"data": cifar100_resnet18_baseline_coslr_e_100_b_128 , "label": "Cifar100 Resnet18 baseline SGD coslr epoch 100"},
    # {"data": cifar100_resnet18_baseline_coslr_e_200_b_128 , "label": "Cifar100 Resnet18 baseline SGD coslr epoch 200"},
    # {"data": cifar100_resnet18_baseline_steplr_e_100_b_128 , "label": "Cifar100 Resnet18 baseline SGD steplr epoch 100"},
    # {"data": cifar100_resnet18_baseline_steplr_e_200_b_128 , "label": "Cifar100 Resnet18 baseline SGD steplr epoch 200"},
    # {"data": cifar100_fdq_resnet18_lr_01_lf_8_hq_2_coslr_e_100_b_128 , "label": "Cifar100 Resnet18 lb 8 hq 2 coslr epoch 100"},
    # {"data": cifar100_fdq_resnet18_lr_01_lf_8_hq_2_coslr_e_200_b_128 , "label": "Cifar100 Resnet18 lb 8 hq 2 coslr epoch 200"},
    # {"data": cifar100_fdq_resnet18_lr_01_lf_8_hq_2_steplr_e_100_b_128 , "label": "Cifar100 Resnet18 lb 8 hq 2 steplr epoch 100"},
    # {"data": cifar100_fdq_resnet18_lr_01_lf_8_hq_2_steplr_e_200_b_128 , "label": "Cifar100 Resnet18 lb 8 hq 2 steplr epoch 200"},

    # # CIFAR 100 Resnet34
    # {"data": cifar100_resnet34_baseline_coslr_e_200_b_128 , "label": "Cifar100 Resnet34 baseline SGD coslr epoch 200 bactchsize 128"},
    # {"data": cifar100_resnet34_baseline_steplr_e_200_b_128 , "label": "Cifar100 Resnet34 baseline SGD steplr epoch 200 bactchsize 128"},
    # {"data": cifar100_fdq_resnet34_lr_01_lf_8_hq_2_coslr_e_200_b_128 , "label": "Cifar100 Resnet34 lb 8 hq 2 coslr epoch 200 bactchsize 128"},
    # {"data": cifar100_fdq_resnet34_lr_01_lf_8_hq_2_steplr_e_200_b_128 , "label": "Cifar100 Resnet34 lb 8 hq 2 steplr epoch 200 bactchsize 128"},
    # {"data": cifar100_resnet34_baseline_coslr_e_200_b_256 , "label": "Cifar100 Resnet34 baseline SGD coslr epoch 200 bactchsize 256"},
    # {"data": cifar100_resnet34_baseline_steplr_e_200_b_256 , "label": "Cifar100 Resnet34 baseline SGD steplr epoch 200 bactchsize 256"},
    # {"data": cifar100_fdq_resnet34_lr_01_lf_8_hq_2_coslr_e_200_b_256 , "label": "Cifar100 Resnet34 lb 8 hq 2 coslr epoch 200 bactchsize 256"},
    # {"data": cifar100_fdq_resnet34_lr_01_lf_8_hq_2_steplr_e_200_b_256 , "label": "Cifar100 Resnet34 lb 8 hq 2 steplr epoch 200 bactchsize 256"},


    # # CIFAR 100 Resnet50
    {"data": cifar100_resnet50_baseline_coslr_e_200_b_128 , "label": "Cifar100 Resnet50 baseline SGD coslr epoch 200"},
    {"data": cifar100_resnet50_baseline_steplr_e_200_b_128 , "label": "Cifar100 Resnet50 baseline SGD steplr epoch 200"},
    {"data": cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_128 , "label": "Cifar100 Resnet50 lb 8 hq 2 coslr epoch 200"},
    {"data": cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_128 , "label": "Cifar100 Resnet50 lb 8 hq 2 steplr epoch 200"},
    {"data": cifar100_resnet50_baseline_coslr_e_200_b_256 , "label": "Cifar100 Resnet50 baseline SGD coslr epoch 200 bactchsize 256"},
    {"data": cifar100_resnet50_baseline_steplr_e_200_b_256 , "label": "Cifar100 Resnet50 baseline SGD steplr epoch 200 bactchsize 256"},
    {"data": cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_256 , "label": "Cifar100 Resnet50 lb 8 hq 2 coslr epoch 200 bactchsize 256"},
    {"data": cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_256 , "label": "Cifar100 Resnet50 lb 8 hq 2 steplr epoch 200 bactchsize 256"},


    # # CIFAR 100 Shufflenet V2 X1.0
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_128 , "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD coslr epoch 200 bactchsize 128"},
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_128 , "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD steplr epoch 200 bactchsize 128"},
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_256 , "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD coslr epoch 200 bactchsize 256"},
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_256 , "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD steplr epoch 200 bactchsize 256"},
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_coslr_e_200_b_1024, "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD coslr epoch 200 bactchsize 1024"},
    # {"data": cifar100_shufflenet_v2_x1_0_baseline_steplr_e_200_b_1024, "label": "Cifar100 Shufflenet_v2_x1_0 baseline SGD steplr epoch 200 bactchsize 1024"},
    # {"data": cifar100_fdq_shufflenet_v2_x1_0_lr_01_lf_8_hq_2_coslr_e_200_b_1024, "label": "Cifar100 Shufflenet_v2_x1_0 lb 8 hq 2 coslr epoch 200 bactchsize 1024"},
    # {"data": cifar100_fdq_shufflenet_v2_x1_0_lr_01_lf_8_hq_2_steplr_e_200_b_1024, "label": "Cifar100 Shufflenet_v2_x1_0 lb 8 hq 2 steplr epoch 200 bactchsize 1024"},

    # # CIFAR 100 Shufflenet V2 X1.0
    # {"data": cifar100_densenet121_baseline_coslr_e_200_b_128 , "label": "Cifar100 Densenet121 baseline SGD coslr epoch 200 bactchsize 128"},
    {"data": cifar100_densenet121_baseline_steplr_e_200_b_128 , "label": "Cifar100 Densenet121 baseline SGD steplr epoch 200 bactchsize 128"},
    # {"data": cifar100_densenet121_baseline_coslr_e_200_b_256 , "label": "Cifar100 Densenet121 baseline SGD coslr epoch 200 bactchsize 256"},
    {"data": cifar100_densenet121_baseline_steplr_e_200_b_256 , "label": "Cifar100 Densenet121 baseline SGD steplr epoch 200 bactchsize 256"},
    {"data": cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_128 , "label": "Cifar100 Densenet121 lb 8 hq 2 steplr epoch 200 bactchsize 128"},
    {"data": cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_256 , "label": "Cifar100 Densenet121 lb 8 hq 2 steplr epoch 200 bactchsize 256"},


    # # ImageNet
    # {"data": imagenet_mobilenet_baseline_sgd_coslr_005_wd_00004, "label": "ImageNet baseline MobileNet-V2 SGD"},
    # # {"data": imagenet_mobilenet_baseline_sgd_lr_005_wd_00004_b, "label": "ImageNet baseline MobileNet-V2 SGD"},
    # # {"data": imagenet_mobilenet_baseline_sgd_lr_01_wd_0001, "label": "ImageNet baseline MobileNet-V2 SGD"},
    # {"data": imagenet_mobilenet_fdq_win_0_1_hfc_2_sgd, "label": "ImageNet fdq win 0.1 hfc 2 MobileNet-V2 SGD"},
    # {"data": imagenet_mobilenet_fdq_win_0_2_hfc_2_sgd, "label": "ImageNet fdq win 0.2 hfc 2 MobileNet-V2 SGD"},
    # {"data": imagenet_mobilenet_fdq_win_0_1_hfc_2_sgd_simulate, "label": "ImageNet fdq win 0.1 hfc 2 MobileNet-V2 SGD Simulate"},
]

for acc_vs_epoch in plot_list:
    plt.plot(acc_vs_epoch["data"], label=acc_vs_epoch["label"])

# plt.xlim([0,100])
# plt.ylim([70,76])
plt.legend()
plt.show()


print(['{:.1f}'.format(idx) for idx in range(10)] + ['{:.0f}'.format(idx)+'.' for idx in range(10, 200)])
for acc_vs_epoch in plot_list:
    print("Max testing accuracy of "+acc_vs_epoch["label"]+" {}".format(np.array(acc_vs_epoch["data"]).max()) + " at epoch " + str(np.array(acc_vs_epoch["data"]).argmax()))

# plt.plot(acc_resnet18_lr_01_win_07)
# plt.plot(acc_resnet18_lr_01_win_08)
# plt.plot(acc_baseline_b, label="baseline SGD")
# plt.plot(acc_baseline_adam, label="baseline Adam")
# plt.plot(dctb_resnet18_lr_01_convwin_07_bnwin_1_sgd, label="conv 0.49 bn 1 SGD")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_1_sgd, label="conv 0.25 bn 1 SGD")
# plt.plot(dctb_resnet18_lr_01_convwin_03_bnwin_1_sgd, label="conv 0.09 bn 1 SGD")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_05_sgd, label="conv 0.25 bn 0.25 SGD")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_05_sgd_b, label="conv 0.25 bn 0.25 SGD")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_1_sgd_auto, label="conv 0.25 bn 1 SGD Auto")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_1_sgd_auto_b, label="conv 0.25 bn 1 SGD Auto")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_1_sgd_auto_c, label="conv 0.25 bn 1 SGD Auto")
# plt.plot(dctb_resnet18_lr_01_convwin_05_bnwin_05_sgd_auto, label="conv 0.25 bn 0.25 SGD Auto")
# plt.plot(acc_baseline_adam, label="baseline Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_07_bnwin_1_adam, label="conv 0.49 bn 1 Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_05_bnwin_1_adam, label="conv 0.25 bn 1 Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_05_bnwin_07_adam, label="conv 0.25, bn 0.49, Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_05_bnwin_05_adam, label="conv 0.25, bn 0.25, Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_05_bnwin_05_adam_b, label="conv 0.25, bn 0.25, Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_03_bnwin_1_adam, label="conv 0.09 bn 1 Adam")
# plt.plot(dctb_resnet18_lr_0001_convwin_02_bnwin_1_adam, label="conv 0.04 bn 1 Adam")







