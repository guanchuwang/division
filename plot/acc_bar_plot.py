import re
import numpy as np



# extract_value('log.txt')

import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# import numpy as np
import json

acc_fp32 = {"cifar10, resnet18": 94.89,
            "cifar10, resnet164": 94.90,
            "cifar100, resnet164": 77.30,
            "cifar100, densenet121": 79.75,
            "cifar10, resnet50": 76.15,
            "cifar10, densenet161": 77.65}

acc_blpa = {"cifar10, resnet18": None,
            "cifar10, resnet164": 94.51,
            "cifar100, resnet164": 76.42,
            "cifar100, densenet121": None,
            "cifar10, resnet50": None,
            "cifar10, densenet161": None}

acc_acgc = {"cifar10, resnet18": 90.43,
            "cifar10, resnet164": None,
            "cifar100, resnet164": 73.37,
            "cifar100, densenet121": None,
            "cifar10, resnet50": 72.27,
            "cifar10, densenet161": None}

acc_actnn = {"cifar10, resnet18": 94.84,
            "cifar10, resnet164": None,
            "cifar100, resnet164": None,
            "cifar100, densenet121": 78.95,
            "cifar10, resnet50": 75.42,
            "cifar10, densenet161": 76.59}

acc_division = {"cifar10, resnet18": 94.72,
            "cifar10, resnet164": 94.50,
            "cifar100, resnet164": 76.9,
            "cifar100, densenet121": 79.47,
            "cifar10, resnet50": 75.86,
            "cifar10, densenet161": 77.58}

bar_width = 5
## DIVISION vs BLPA

fig = plt.figure(figsize=(5.5, 4.5))
acc_fp32_buf = []
acc_blpa_buf = []
acc_division_buf = []
for key, value in acc_fp32.items():

    if acc_blpa[key] is not None:
        acc_fp32_buf.append(value)
        acc_blpa_buf.append(acc_blpa[key])
        acc_division_buf.append(acc_division[key])
        # print(key, value)
        # print(key, acc_blpa[key])
        # print(key, acc_division[key])

acc_fp32_text_buf = [str(round(x*10)/10) for x in acc_fp32_buf]
acc_blpa_text_buf = [str(round(x*10)/10) for x in acc_blpa_buf]
acc_division_text_buf = [str(round(x*10)/10) for x in acc_division_buf]

division_axis = np.arange(len(acc_fp32_buf)) * 4 * bar_width + bar_width
vanilla_axis = division_axis - bar_width
blpa_axis = division_axis + bar_width

plt.bar(vanilla_axis, acc_fp32_buf, width=bar_width, color="b", edgecolor="black", label="Normal")
plt.bar(division_axis, acc_division_buf, width=bar_width, color="r", edgecolor="black", label="DIVISION")
plt.bar(blpa_axis, acc_blpa_buf, width=bar_width, color="darkgoldenrod", edgecolor="black", label="BLPA")

for idx in range(len(acc_fp32_buf)):
    plt.text(vanilla_axis[idx] - 0.5*bar_width, acc_fp32_buf[idx], acc_fp32_text_buf[idx], size=18, color="b")
    plt.text(blpa_axis[idx] - 0.5*bar_width, acc_blpa_buf[idx], acc_blpa_text_buf[idx], size=18, color="darkgoldenrod")
    plt.text(division_axis[idx] - 0.5*bar_width, acc_division_buf[idx], acc_division_text_buf[idx], size=18, color="r")


xstr = ["CIFAR-10 \n ResNet-164", "CIFAR-100 \n ResNet-164"]
plt.xticks(division_axis, xstr, fontsize=25)

plt.xlim([division_axis[0] - 2*bar_width, division_axis[-1] + 2*bar_width])
plt.ylim([50, 100])
plt.ylabel("Accuracy", fontsize=25)
plt.legend(loc='upper right', fontsize=18, ncol=1)
plt.grid(axis='y')
plt.yticks(fontsize=25)
plt.subplots_adjust(left=0.21, bottom=0.17, top=0.97, right=0.99)
# plt.savefig("../figure/acc_vs_blpa.pdf")
plt.savefig("../figure/acc_vs_blpa.png")
# plt.show()
plt.close()


## DIVISION vs AC-GC
fig = plt.figure(figsize=(7.5, 4.5))
acc_fp32_buf = []
acc_acgc_buf = []
acc_division_buf = []
for key, value in acc_fp32.items():

    if acc_acgc[key] is not None:
        acc_fp32_buf.append(value)
        acc_acgc_buf.append(acc_acgc[key])
        acc_division_buf.append(acc_division[key])
        # print(key, value)
        # print(key, acc_acgc[key])
        # print(key, acc_division[key])

acc_fp32_text_buf = [str(round(x*10)/10) for x in acc_fp32_buf]
acc_acgc_text_buf = [str(round(x*10)/10) for x in acc_acgc_buf]
acc_division_text_buf = [str(round(x*10)/10) for x in acc_division_buf]

division_axis = np.arange(len(acc_fp32_buf)) * 4 * bar_width + bar_width
vanilla_axis = division_axis - bar_width
acgc_axis = division_axis + bar_width

plt.bar(vanilla_axis, acc_fp32_buf, width=bar_width, color="b", edgecolor="black", label="Normal")
plt.bar(division_axis, acc_division_buf, width=bar_width, color="r", edgecolor="black", label="DIVISION")
plt.bar(acgc_axis, acc_acgc_buf, width=bar_width, color="darkmagenta", edgecolor="black", label="AC-GC")

for idx in range(len(acc_fp32_buf)):
    plt.text(vanilla_axis[idx] - 0.5*bar_width, acc_fp32_buf[idx], acc_fp32_text_buf[idx], size=18, color="b")
    plt.text(acgc_axis[idx] - 0.5*bar_width, acc_acgc_buf[idx], acc_acgc_text_buf[idx], size=18, color="darkmagenta")
    plt.text(division_axis[idx] - 0.5*bar_width, acc_division_buf[idx], acc_division_text_buf[idx], size=18, color="r")


xstr = ["CIFAR-10 \n ResNet-18", "CIFAR-100 \n ResNet-164", "ImageNet \n ResNet-50"]
plt.xticks(division_axis, xstr, fontsize=25)

plt.xlim([division_axis[0] - 2*bar_width, division_axis[-1] + 2*bar_width])
plt.ylim([50, 100])
# plt.ylabel("Accuracy", fontsize=25)
plt.legend(loc='upper right', fontsize=18, ncol=1)
plt.grid(axis='y')
plt.yticks(fontsize=25)
plt.subplots_adjust(left=0.1, bottom=0.17, top=0.97, right=0.99)
# plt.savefig("../figure/acc_vs_acgc.pdf")
plt.savefig("../figure/acc_vs_acgc.png")
# plt.show()
plt.close()


## DIVISION vs ActNN
fig = plt.figure(figsize=(10, 4.5))
acc_fp32_buf = []
acc_actnn_buf = []
acc_division_buf = []
for key, value in acc_fp32.items():

    if acc_actnn[key] is not None:
        acc_fp32_buf.append(value)
        acc_actnn_buf.append(acc_actnn[key])
        acc_division_buf.append(acc_division[key])
        # print(key, value)
        # print(key, acc_actnn[key])
        # print(key, acc_division[key])

acc_fp32_text_buf = [str(round(x*10)/10) for x in acc_fp32_buf]
acc_actnn_text_buf = [str(round(x*10)/10) for x in acc_actnn_buf]
acc_division_text_buf = [str(round(x*10)/10) for x in acc_division_buf]

division_axis = np.arange(len(acc_fp32_buf)) * 4 * bar_width + bar_width
vanilla_axis = division_axis - bar_width
actnn_axis = division_axis + bar_width

plt.bar(vanilla_axis, acc_fp32_buf, width=bar_width, color="b", edgecolor="black", label="Normal")
plt.bar(division_axis, acc_division_buf, width=bar_width, color="r", edgecolor="black", label="DIVISION")
plt.bar(actnn_axis, acc_actnn_buf, width=bar_width, color="g", edgecolor="black", label="ActNN")

for idx in range(len(acc_fp32_buf)):
    plt.text(vanilla_axis[idx] - 0.5*bar_width, acc_fp32_buf[idx], acc_fp32_text_buf[idx], size=18, color="b")
    plt.text(actnn_axis[idx] - 0.5*bar_width, acc_actnn_buf[idx], acc_actnn_text_buf[idx], size=18, color="g")
    plt.text(division_axis[idx] - 0.5*bar_width, acc_division_buf[idx], acc_division_text_buf[idx], size=18, color="r")


xstr = ["CIFAR-10 \n ResNet-18", "CIFAR-100 \n DenseNet-121", "ImageNet \n ResNet-50", "ImageNet \n DenseNet-161"]
plt.xticks(division_axis, xstr, fontsize=25)

plt.xlim([division_axis[0] - 2*bar_width, division_axis[-1] + 2*bar_width])
plt.ylim([50, 100])
# plt.ylabel("Accuracy", fontsize=25)
plt.legend(loc='upper right', fontsize=18, ncol=1)
plt.grid(axis='y')
plt.yticks(fontsize=25)
plt.subplots_adjust(left=0.07, bottom=0.17, top=0.97, right=0.98)
# plt.savefig("../figure/acc_vs_actnn.pdf")
plt.savefig("../figure/acc_vs_actnn.png")
# plt.show()
plt.close()

