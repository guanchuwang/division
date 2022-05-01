import re

def extract_value(fname, regex_str, strloc, endloc):
    f = open(fname, 'r')
    str_buf = f.read()

    regex = re.compile(regex_str)

    str_list = regex.findall(str_buf)
    print(str_list)
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
import torch
import seaborn as sns
import numpy as np

epoch = extract_value('log/QAT-20210715-030212/log.txt', 'epoch : .+; l', 8, -3)
loss_value = extract_value('log/QAT-20210715-030212/log.txt', 'loss : {.+} ; a', 8, -5)

plt.plot(epoch, loss_value)
plt.show()


