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

def average_max(x, milestone_cifar10):

    # print(x)
    z = np.zeros_like(milestone_cifar10).astype(np.float)
    for index, epoch in enumerate(milestone_cifar10):
        # print(epoch)
        # print(x[0:epoch])
        z[index] = x[0:epoch].max()
    return z



# extract_value('log.txt')

import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# import numpy as np


acc_baseline_c = extract_value('../log/resnet18_cifar10_baseline.log', 'Acc@1 .+ Acc@5', 6, -6)
fdq_resnet18_lr_01_lf_8_hq_2 = extract_value('../log/resnet18_cifar10_lb_8_hq_2.log', 'Acc@1 .+ Acc@5', 6, -6)

cifar100_resnet50_baseline_coslr_e_200_b_128 = extract_value('../resnet50_cifar100_baseline_coslr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_steplr_e_200_b_128 = extract_value('../resnet50_cifar100_baseline_steplr_e_200.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_128 = extract_value('../resnet50_cifar100_lb_8_hq_2_coslr_e_200_b_128_s_7.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../resnet50_cifar100_lb_8_hq_2_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_coslr_e_200_b_256 = extract_value('../resnet50_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_resnet50_baseline_steplr_e_200_b_256 = extract_value('../resnet50_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_coslr_e_200_b_256 = extract_value('../resnet50_cifar100_lb_8_hq_2_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_fdq_resnet50_lr_01_lf_8_hq_2_steplr_e_200_b_256 = extract_value('../resnet50_cifar100_lb_8_hq_2_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)

cifar100_densenet121_baseline_coslr_e_200_b_128 = extract_value('../densenet121_cifar100_baseline_coslr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_steplr_e_200_b_128 = extract_value('../densenet121_cifar100_baseline_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_coslr_e_200_b_256 = extract_value('../densenet121_cifar100_baseline_coslr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_baseline_steplr_e_200_b_256 = extract_value('../densenet121_cifar100_baseline_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_128 = extract_value('../densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_128.log', 'Acc@1 .+ Acc@5', 6, -6)
cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_256 = extract_value('../densenet121_cifar100_lb_8_hq_2_steplr_e_200_b_256.log', 'Acc@1 .+ Acc@5', 6, -6)

bar_width = 0.2
epoch_cifar10 = 50

milestone_cifar10 = np.arange(5, epoch_cifar10+1, 5)
x_label_cifar10 = [str(epoch_index) for epoch_index in milestone_cifar10]
acc_vanilla_cifar10 = average_max(np.array(acc_baseline_c), milestone_cifar10)
acc_Division_cifar10 = average_max(np.array(fdq_resnet18_lr_01_lf_8_hq_2), milestone_cifar10)

print(acc_vanilla_cifar10)
print(acc_Division_cifar10)

vanilla_axis = np.arange(len(x_label_cifar10)) * 6 * bar_width
Division_axis = vanilla_axis + bar_width
# hfc_axis = lfc_axis + bar_width
# baseline_axis = lfc_axis - bar_width

plt.bar(vanilla_axis, acc_vanilla_cifar10, width=bar_width, color="b", label="Vanilla")
plt.bar(Division_axis, acc_Division_cifar10, width=bar_width, color="r", label="Division")

plt.xticks(vanilla_axis + bar_width, x_label_cifar10, fontsize=20)
plt.yticks(fontsize=20)
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# plt.gca().ticklabel_format(style='sci', scilimits=(0,2), axis='y')
plt.ylim([40, 95])
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Val. Accuracy (%)", fontsize=20)
plt.legend(loc='upper left', fontsize=20)

plt.subplots_adjust(left=0.12, bottom=0.14, top=0.99, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)
plt.savefig("../figure/acc_converge_cifar10.pdf")

# plt.show()
plt.close()
#

epoch_cifar100 = 200

milestone_cifar100 = np.arange(20, epoch_cifar100+1, 20)
x_label_cifar100 = [str(epoch_index) for epoch_index in milestone_cifar100]
acc_vanilla_cifar100 = average_max(np.array(cifar100_densenet121_baseline_steplr_e_200_b_128), milestone_cifar100)
acc_Division_cifar100 = average_max(np.array(cifar100_densenet121_lr_01_lf_8_hq_2_steplr_e_200_b_128), milestone_cifar100)

print(acc_vanilla_cifar100)
print(acc_Division_cifar100)

vanilla_axis = np.arange(len(x_label_cifar100)) * 6 * bar_width
Division_axis = vanilla_axis + bar_width
# hfc_axis = lfc_axis + bar_width
# baseline_axis = lfc_axis - bar_width

plt.bar(vanilla_axis, acc_vanilla_cifar100, width=bar_width, color="b", label="Vanilla")
plt.bar(Division_axis, acc_Division_cifar100, width=bar_width, color="r", label="Division")

plt.xticks(vanilla_axis + bar_width, x_label_cifar100, fontsize=20)
plt.yticks(fontsize=20)
# plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# plt.gca().ticklabel_format(style='sci', scilimits=(0,2), axis='y')
plt.ylim([30, 80])
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Val. Accuracy (%)", fontsize=20)
plt.legend(loc='upper left', fontsize=20)

plt.subplots_adjust(left=0.12, bottom=0.14 , top=0.985, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)
plt.savefig("../figure/acc_converge_cifar100.pdf")

# plt.show()
plt.close()

