import numpy as np
import pdb
import matplotlib.pyplot as plt
# import seaborn as sns # improves plot aesthetics

FONTSIZE = 14
LABELSIZE = 15
LEGENDSIZE = 14
TEXTSIZE = 20
font_config = {"font.size": FONTSIZE} # , "font.family": "Times New Roman"
plt.rcParams.update(font_config)

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])
def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=4):
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True,
                label = "axes{}".format(i)) for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables, fontsize=25, position=(0.0, -0.25))
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                # grid = grid[::-1] # hack to invert grid
                          # gridlabels aren"t reversed
                pass
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i], fontsize=25)
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# example data
blank = " "
variables = ("\n\n Acc (%) \n ImageNet", "Compression Rate \n ImageNet", "Throughput \n ImageNet", "\n Acc (%) \n CIFAR-100", "Compression Rate \n CIFAR-100", "Throughput \n CIFAR-100")
# variables = ("1", "2", "2", "2", "2", "2")
datas = ([75.9, 10.4, 380.72, 76.3, 6.44, 2873], [75.4, 10.6, 308.33, 75.7, 7, 2199], [76.2, 3.58, 378.22, 77.1, 2.76, 2702])
ranges = [(0, 80.00), (0, 11), (250, 390), (0, 80.00), (0, 7), (250, 3000)]
zorder = [3, 2, 1]
# datas = ([79.03, 573, .20], [79.13, 82, .31])
# ranges = [(78.90, 79.20), (650, 50), (0.0, 0.5)]
# plotting
color_buf = ["red", "green", "darkmagenta", ]
fig1 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig1, variables, ranges)
for idx, data in enumerate(datas):
    radar.plot(data, color=color_buf[idx], linewidth=6, markersize=16, marker="o", zorder=zorder[idx])
    radar.fill(data, alpha=0.2)

radar.ax.legend(["DIVISION", "ActNN", "Checkpoint"],
                labelspacing=0.1, fontsize=25, loc=(1.2, 0.6), frameon=False)
# plt.subplots_adjust(left=0.01, bottom=0.01, top=0.99, right=0.99, wspace=0.01) # (left=0.125, bottom=0.155, top=0.965, right=0.97, wspace=0.01)

plt.savefig(r"../figure/radar.png", bbox_inches="tight", dpi=50)
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# # name = ["Normal", "ActNN", "Checkpoint", "SWAP", "DIVISION"]
# eval_metric = ["Accuracy", "Compression Rate", "Throughput"]
# theta = np.linspace(0, 2*np.pi, len(eval_metric), endpoint=False)
# value = np.random.randint(50,100, size=3)
# theta = np.concatenate((theta, [theta[0]]))
# value = np.concatenate((value, [value[0]]))
#
# Ymax = 100
# normal_training = np.array([76.2, 1, 469.05, 77.6])/np.array([80, 12, 400, 80]) * Ymax
# ActNN_training = np.array([75.4, 10.6, 308.33, 75.4])/np.array([80, 12, 400, 80]) * Ymax
# Checkpoint_training = np.array([76.2, 3.58, 122.14, 76.2])/np.array([80, 12, 400, 80]) * Ymax
# SWAP_training = np.array([76.2, 1, 53.89, 76.2])/np.array([80, 12, 400, 80]) * Ymax
# DIVISION_training = np.array([75.9, 10.4, 380.72, 75.9])/np.array([80, 12, 400, 80]) * Ymax
#
# ax = plt.subplot(111,projection = 'polar')
# # ax.plot(theta, normal_training,'m-', lw=1, alpha=0.2)
# # ax.fill(theta, normal_training,'m', alpha=0.2)
# ax.plot(theta, SWAP_training, color="darkmagenta", lw=3, alpha=0.9, markersize=10, marker="o")
# ax.fill(theta, SWAP_training, color="darkmagenta", alpha=0.2)
# ax.plot(theta, Checkpoint_training, color="darkgoldenrod", lw=3, alpha=0.9, markersize=10, marker="o")
# ax.fill(theta, Checkpoint_training, color="darkgoldenrod", alpha=0.2)
# ax.plot(theta, ActNN_training, color="green", lw=3, alpha=0.9, markersize=10, marker="o")
# ax.fill(theta, ActNN_training, color="green", alpha=0.2)
# ax.plot(theta, DIVISION_training, color="red", lw=3, alpha=0.9, markersize=10, marker="o")
# ax.fill(theta, DIVISION_training, color="red", alpha=0.2)
# ax.set_thetagrids(theta[0:-1]*180/np.pi, eval_metric, fontsize=18)
# # ax.set_ylim(0, Ymax)
# ax.set_rlim(0, Ymax)
# ax.set_yticks(["a"])
# ax.set_theta_zero_location('N')
# # ax.set_title('f',fontsize = 20)
# plt.show()