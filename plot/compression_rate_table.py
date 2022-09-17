import re
import numpy as np



# extract_value('log.txt')

import matplotlib.pyplot as plt
# import torch
# import seaborn as sns
# import numpy as np
import json

with open("../log/memory/memory_debug_resnet50_division_b64.json") as json_file:
    memory_resnet_division_b64 = json.load(json_file)
with open("../log/memory/memory_debug_resnet50_division_b128.json") as json_file:
    memory_resnet_division_b128 = json.load(json_file)
with open("../log/memory/memory_debug_resnet50_division_b256.json") as json_file:
    memory_resnet_division_b256 = json.load(json_file)
with open("../log/memory/memory_debug_resnet50_division_b512.json") as json_file:
    memory_resnet_division_b512 = json.load(json_file)

 
with open("../log/memory/baselines/memory_debug_resnet50_actnn_b64.json") as json_file:
    memory_resnet_actnn_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_actnn_b128.json") as json_file:
    memory_resnet_actnn_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_actnn_b256.json") as json_file:
    memory_resnet_actnn_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_actnn_b512.json") as json_file:
    memory_resnet_actnn_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_resnet50_BLPA_b64.json") as json_file:
    memory_resnet_BLPA_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_BLPA_b128.json") as json_file:
    memory_resnet_BLPA_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_BLPA_b256.json") as json_file:
    memory_resnet_BLPA_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_BLPA_b512.json") as json_file:
    memory_resnet_BLPA_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_resnet50_ACGC_b64.json") as json_file:
    memory_resnet_ACGC_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_ACGC_b128.json") as json_file:
    memory_resnet_ACGC_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_ACGC_b256.json") as json_file:
    memory_resnet_ACGC_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_ACGC_b512.json") as json_file:
    memory_resnet_ACGC_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_resnet50_checkpoint_b64.json") as json_file:
    memory_resnet_checkpoint_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_checkpoint_b128.json") as json_file:
    memory_resnet_checkpoint_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_checkpoint_b256.json") as json_file:
    memory_resnet_checkpoint_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_checkpoint_b512.json") as json_file:
    memory_resnet_checkpoint_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_resnet50_vanilla_b64.json") as json_file:
    memory_resnet_vanilla_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_vanilla_b128.json") as json_file:
    memory_resnet_vanilla_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_resnet50_vanilla_b256.json") as json_file:
    memory_resnet_vanilla_b256 = json.load(json_file)


with open("../log/memory/memory_debug_wrn50_2_division_b64.json") as json_file:
    memory_wrn_division_b64 = json.load(json_file)
with open("../log/memory/memory_debug_wrn50_2_division_b128.json") as json_file:
    memory_wrn_division_b128 = json.load(json_file)
with open("../log/memory/memory_debug_wrn50_2_division_b256.json") as json_file:
    memory_wrn_division_b256 = json.load(json_file)
with open("../log/memory/memory_debug_wrn50_2_division_b512.json") as json_file:
    memory_wrn_division_b512 = json.load(json_file)


with open("../log/memory/baselines/memory_debug_wrn50_2_actnn_b64.json") as json_file:
    memory_wrn_actnn_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_actnn_b128.json") as json_file:
    memory_wrn_actnn_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_actnn_b256.json") as json_file:
    memory_wrn_actnn_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_actnn_b512.json") as json_file:
    memory_wrn_actnn_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_wrn50_2_BLPA_b64.json") as json_file:
    memory_wrn_BLPA_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_BLPA_b128.json") as json_file:
    memory_wrn_BLPA_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_BLPA_b256.json") as json_file:
    memory_wrn_BLPA_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_BLPA_b512.json") as json_file:
    memory_wrn_BLPA_b512 = json.load(json_file)

with open("../log/memory/baselines/memory_debug_wrn50_2_ACGC_b64.json") as json_file:
    memory_wrn_ACGC_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_ACGC_b128.json") as json_file:
    memory_wrn_ACGC_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_ACGC_b256.json") as json_file:
    memory_wrn_ACGC_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_ACGC_b512.json") as json_file:
    memory_wrn_ACGC_b512 = json.load(json_file)
#
with open("../log/memory/baselines/memory_debug_wrn50_2_checkpoint_b64.json") as json_file:
    memory_wrn_checkpoint_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_checkpoint_b128.json") as json_file:
    memory_wrn_checkpoint_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_checkpoint_b256.json") as json_file:
    memory_wrn_checkpoint_b256 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_checkpoint_b512.json") as json_file:
    memory_wrn_checkpoint_b512 = json.load(json_file)
#
with open("../log/memory/baselines/memory_debug_wrn50_2_vanilla_b64.json") as json_file:
    memory_wrn_vanilla_b64 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_vanilla_b128.json") as json_file:
    memory_wrn_vanilla_b128 = json.load(json_file)
with open("../log/memory/baselines/memory_debug_wrn50_2_vanilla_b256.json") as json_file:
    memory_wrn_vanilla_b256 = json.load(json_file)


bactch_size_buf = np.array([64, 128, 256, 512])
# epoch_cifar10 = 50

memory_index_buf = ["total", "activation"]
mem_compression_rate_rate_buf_buf = []
for mem_idx in memory_index_buf:

    memory_resnet50_division = [# memory_resnet_division_b32[mem_idx],
                               memory_resnet_division_b64[mem_idx],
                               memory_resnet_division_b128[mem_idx],
                               memory_resnet_division_b256[mem_idx],
                               memory_resnet_division_b512[mem_idx]
                               ]

    memory_resnet50_actnn = [# memory_resnet_actnn_b32[mem_idx],
                            memory_resnet_actnn_b64[mem_idx],
                            memory_resnet_actnn_b128[mem_idx],
                            memory_resnet_actnn_b256[mem_idx],
                            memory_resnet_actnn_b512[mem_idx]
                               ]

    memory_resnet50_BLPA = [# memory_resnet_BLPA_b32[mem_idx],
                           memory_resnet_BLPA_b64[mem_idx],
                           memory_resnet_BLPA_b128[mem_idx],
                           memory_resnet_BLPA_b256[mem_idx],
                           memory_resnet_BLPA_b512[mem_idx]
                               ]

    memory_resnet50_ACGC = [  # memory_resnet_ACGC_b32[mem_idx],
                            memory_resnet_ACGC_b64[mem_idx],
                            memory_resnet_ACGC_b128[mem_idx],
                            memory_resnet_ACGC_b256[mem_idx],
                            memory_resnet_ACGC_b512[mem_idx]
                        ]

    memory_resnet50_checkpoint = [# memory_resnet_checkpoint_b32[mem_idx],
                                 memory_resnet_checkpoint_b64[mem_idx],
                                 memory_resnet_checkpoint_b128[mem_idx],
                                 memory_resnet_checkpoint_b256[mem_idx],
                                 memory_resnet_checkpoint_b512[mem_idx]
                               ]

    memory_resnet50_vanilla = [ # memory_resnet_vanilla_b32[mem_idx],
                              memory_resnet_vanilla_b64[mem_idx],
                              memory_resnet_vanilla_b128[mem_idx],
                              memory_resnet_vanilla_b256[mem_idx],
                              # memory_resnet_vanilla_b512[mem_idx]
                              ]

    memory_wrn_division = [# memory_wrn_division_b32[mem_idx],
                           memory_wrn_division_b64[mem_idx],
                           memory_wrn_division_b128[mem_idx],
                           memory_wrn_division_b256[mem_idx],
                           memory_wrn_division_b512[mem_idx]
                           ]

    memory_wrn_actnn = [# memory_wrn_actnn_b32[mem_idx],
                        memory_wrn_actnn_b64[mem_idx],
                        memory_wrn_actnn_b128[mem_idx],
                        memory_wrn_actnn_b256[mem_idx],
                        memory_wrn_actnn_b512[mem_idx]
                           ]

    memory_wrn_BLPA = [ # memory_wrn_BLPA_b32[mem_idx],
                           memory_wrn_BLPA_b64[mem_idx],
                           memory_wrn_BLPA_b128[mem_idx],
                           memory_wrn_BLPA_b256[mem_idx],
                           memory_wrn_BLPA_b512[mem_idx]
                               ]

    memory_wrn_ACGC = [  # memory_wrn_ACGC_b32[mem_idx],
                        memory_wrn_ACGC_b64[mem_idx],
                        memory_wrn_ACGC_b128[mem_idx],
                        memory_wrn_ACGC_b256[mem_idx],
                        memory_wrn_ACGC_b512[mem_idx]
                    ]
    #
    memory_wrn_checkpoint = [ # memory_wrn_checkpoint_b32[mem_idx],
                             memory_wrn_checkpoint_b64[mem_idx],
                             memory_wrn_checkpoint_b128[mem_idx],
                             memory_wrn_checkpoint_b256[mem_idx],
                             memory_wrn_checkpoint_b512[mem_idx]
                           ]
    #
    memory_wrn_vanilla = [ # memory_wrn_vanilla_b32[mem_idx],
                              memory_wrn_vanilla_b64[mem_idx],
                              memory_wrn_vanilla_b128[mem_idx],
                              memory_wrn_vanilla_b256[mem_idx],
                              # memory_wrn_vanilla_b512[mem_idx]
                              ]

    memory_buf = tuple(memory_resnet50_vanilla) + tuple(memory_wrn_vanilla) + \
                    tuple(memory_resnet50_vanilla) + tuple(memory_wrn_vanilla) + \
                    tuple(memory_resnet50_checkpoint) + tuple(memory_wrn_checkpoint) + \
                    tuple(memory_resnet50_BLPA) + tuple(memory_wrn_BLPA) + \
                    tuple(memory_resnet50_ACGC) + tuple(memory_wrn_ACGC) + \
                    tuple(memory_resnet50_actnn) + tuple(memory_wrn_actnn) + \
                    tuple(memory_resnet50_division) + tuple(memory_wrn_division)

    compression_rate_resnet50_swap = (1, 1, 1)
    compression_rate_resnet50_checkpoint = tuple([memory_resnet50_vanilla[idx]/memory_resnet50_checkpoint[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_resnet50_BLPA = tuple([memory_resnet50_vanilla[idx]/memory_resnet50_BLPA[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_resnet50_ACGC = tuple([memory_resnet50_vanilla[idx]/memory_resnet50_ACGC[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_resnet50_actnn = tuple([memory_resnet50_vanilla[idx]/memory_resnet50_actnn[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_resnet50_division = tuple([memory_resnet50_vanilla[idx]/memory_resnet50_division[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_wrn_swap = (1, 1, 1)
    compression_rate_wrn_checkpoint = tuple([memory_wrn_vanilla[idx]/memory_wrn_checkpoint[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_wrn_BLPA = tuple([memory_wrn_vanilla[idx]/memory_wrn_BLPA[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_wrn_ACGC = tuple([memory_wrn_vanilla[idx]/memory_wrn_ACGC[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_wrn_actnn = tuple([memory_wrn_vanilla[idx]/memory_wrn_actnn[idx] for idx in range(len(memory_resnet50_vanilla))])
    compression_rate_wrn_division = tuple([memory_wrn_vanilla[idx]/memory_wrn_division[idx] for idx in range(len(memory_resnet50_vanilla))])

    compression_rate_buf = compression_rate_resnet50_checkpoint + compression_rate_wrn_checkpoint + \
                            compression_rate_resnet50_BLPA + compression_rate_wrn_BLPA + \
                            compression_rate_resnet50_ACGC + compression_rate_wrn_ACGC + \
                            compression_rate_resnet50_actnn + compression_rate_wrn_actnn + \
                            compression_rate_resnet50_division + compression_rate_wrn_division

    print(len(memory_buf))
    print(len(compression_rate_buf))

    mem_compression_rate_rate_buf = []
    index2 = 0
    for index in range(len(memory_buf)):
        mem_compression_rate_rate_buf.append(memory_buf[index])
        if index >= 4*(len(bactch_size_buf)-1):
            if index % len(bactch_size_buf) == len(bactch_size_buf) - 1:
                pass
            else:
                # print(index, index2, len(compression_rate_buf), len(memory_buf))
                mem_compression_rate_rate_buf.append(compression_rate_buf[index2])
                index2 += 1

    # print(mem_compression_rate_rate_buf)
    print(len(mem_compression_rate_rate_buf))

    mem_compression_rate_rate_buf_buf.extend(mem_compression_rate_rate_buf)

table = r'''\begin{tabular}{c|c|cccc|cccc}
    \hline
    \hline
         \multicolumn{2}{c|}{Architecture} &  \multicolumn{4}{c|}{ResNet-50} & \multicolumn{4}{c}{WRN-50-2} \\
    \hline
        \multicolumn{2}{c|}{Batch-size} & 64 & 128 & 256 & 512 & 64 & 128 & 256 & 512 \\
    \hline
        \multirow{7}{*}{\makecell[c]{\!\!\!\!Total. Mem\!\!\!\! \\ (GB)}}
        & Vanilla & %.2f & %.2f & %.2f & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! & %.2f & %.2f & %.2f & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & SWAP & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! &  \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & \!\!\!\!Checkpoint\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f\!\!\!\! \\
        \cline{2-10}

        & BLPA & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! \\
        \cline{2-10}
        
        & AC-GC & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & ActNN & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \!\!\!\!\!\!\!\!\textbf{%.2f}\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & \!\!\!\!\Algnameabbr{}\!\!\!\! & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \!\!\!\!\!\!\!\!\textbf{%.2f}\!\!\!\!\!\!\!\! \\

        \hline

        \multirow{7}{*}{\makecell[c]{\!\!\!\!Act. Mem \!\!\!\! \\ (GB)}}
        & Vanilla & %.2f & %.2f & %.2f & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! & %.2f & %.2f & %.2f & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & SWAP & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! &  \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(1\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!\!\!OOM\!\!\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & \!\!\!\!Checkpoint\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f\!\!\!\! \\
        \cline{2-10}

        & BLPA & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! \\
        \cline{2-10}
        
        & AC-GC & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\! & \!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & ActNN & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \!\!\!\!\!\!\!\!\textbf{%.2f}\!\!\!\!\!\!\!\! \\
        \cline{2-10}

        & \!\!\!\!\Algnameabbr{}\!\!\!\! & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!\!\!\!\!%.2f\!\!\!\!\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \textbf{\!\!\!\!%.2f~(%.1f\!$\times$\!)\!\!\!\!} & \!\!\!\!\!\!\!\!\textbf{%.2f}\!\!\!\!\!\!\!\! \\
    \hline
    \hline
    \end{tabular}''' % tuple(mem_compression_rate_rate_buf_buf)

print(table)

with open("table3.txt", "w") as f:
    f.write(table)
f.close()


