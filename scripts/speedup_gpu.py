from pprint import pprint
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

def add_value_labels(ax, spacing=1, upper=5.1, lower=0.0):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        
        
        if y_value < upper and y_value > 0.8:
            continue
        y = y_value
        if y_value > upper:
            y = upper
        if y_value < lower:
            y = lower
        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        # if y_value == 0:
        #    continue
        
        if y_value > lower and y_value <= 0.8:
            ax.annotate(
                label,                # Use `label` as label
                (x_value, y),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                fontsize=5,
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
            continue
        
        if y_value <= lower:
            ax.annotate(
                "×",                      # Use `label` as label
                (x_value, y),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                fontsize=9,
                color = 'red',
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
        
        else:
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                fontsize=9,
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.
        

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

# prepare data
redundancy = {}
labels = []
ANSOR = []
FAMILYSEER = []
FAMILYSEER_GPU = []
FAMILYSEER_GPU_PARALLEL = []
AUTOTVM = []
f = open("speedup_gpu.csv")
for line in f:
    div = line.split(',')
    labels.append(div[0])
    ANSOR.append(float(div[1]))
    FAMILYSEER.append(float(div[2]))
    #FAMILYSEER_GPU.append(float(div[3]))
    #FAMILYSEER_GPU_PARALLEL.append(float(div[4]))
    #AUTOTVM.append(float(div[5]))
# print('======== loaded ========')
# print(redundancy)

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

# brewer2mpl.get_map args: set name set type number of colors
# bmap = brewer2mpl.get_map('RdYlBu', 'diverging', 5)

mycolors = [ '#3399CC', '#003366', '#CCCCCC','#666666', '#ffe138' ]
#mycolors = [ '#fc8d59', '#99d594' ]

# # set color
#plt.rcParams['axes.color_cycle'] = mycolors
# print(plt.style.available)
# plt.style.use('seaborn-darkgrid')

# Plot configuration
f, ax1 = plt.subplots(1, 1, figsize=(12, 2))
plt.xticks(rotation=320,fontsize=12)
# make the figure tight
ax1.set_xlim(-1,len(labels)*2-1)
xlabels = [i for i in range(0,len(labels)*2,2)]
ax1.set_xticks(xlabels)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.8,5.1)

ylabels = [i for i in range(0, 6)]
ax1.set_yticks(ylabels)
ax1.set_yticklabels(ylabels)

# ax1.set_xlabel('Cases')
ax1.set_ylabel('Speedup',fontsize=14)
font = {'style':'normal','weight':'bold'}

#x = [ i for i in range(1, len(labels))]
#x = [i * 3 for i in x]


#x1= x
width = 0.4
if len(AUTOTVM) > 0:
    bar5 = ax1.bar([i-2*width for i in xlabels], AUTOTVM, color=mycolors[4], width=width, label='AutoTVM', edgecolor='black', linewidth=0.1,zorder=12)
if len(ANSOR) > 0:
    bar1 = ax1.bar([i-1*width for i in xlabels], ANSOR, color=mycolors[3], width=width, label='Ansor', edgecolor='black', linewidth=0.1,zorder=10)
if len(FAMILYSEER) > 0:
    bar2 = ax1.bar([i for i in xlabels], FAMILYSEER, color=mycolors[2], width=width, label='FamilySeer', edgecolor='black', linewidth=0.1,zorder=4)
if len(FAMILYSEER_GPU) > 0:
    bar3 = ax1.bar([i+1*width for i in xlabels], FAMILYSEER_GPU, color=mycolors[1], width=width, label='FamilySeer+GPU', edgecolor='black', linewidth=0.1,zorder=5)
if len(FAMILYSEER_GPU_PARALLEL) > 0:
    bar4 = ax1.bar([i+2*width for i in xlabels], FAMILYSEER_GPU_PARALLEL, color=mycolors[0], width=width, label='FamilySeer+GPU+PARALLEL', edgecolor='black', linewidth=0.1,zorder=5)
add_value_labels(ax1)
#ax1.annotate(s='',xy=(-0.5,0),xytext=(-0.5,-32),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
offset=-1
#X轴竖线
#ax1.annotate(s='',xy=(offset,0.0),xytext=(offset,-0.65),arrowprops=dict(facecolor='black', headlength=2.0, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"ResNet50_v1",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=2.0, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"ResNet152_v2",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"Mobilenet",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"Mobilenetv2",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"ViT-H*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"BERT-L*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"RoBERTa-L*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+6
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-1.8),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+3,-2.0,"GPT2-S*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

#offset=offset+6
#ax1.annotate(s='',xy=(offset,0.0),xytext=(offset,-0.4),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
#ax1.text(offset+3,-0.5,"Average",fontdict=font,verticalalignment="bottom",horizontalalignment="center")

ax1.legend(loc='upper right', ncol=5)
f.savefig('figure-6.pdf', bbox_inches = 'tight')
plt.show()