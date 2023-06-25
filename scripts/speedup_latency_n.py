from pprint import pprint
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

def add_value_labels(ax, spacing=1, upper=3, lower=0.0):
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

        if y_value < upper and y_value > lower:
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
        
        if y_value == 0:
            continue
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            fontsize=10,
            rotation=270,
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

# prepare data
redundancy = {}
labels = []
PYTORCH = []
AUTOTVM = []
ANSOR = []
FAMILYSEER = []
f = open("speedup_latency_n.csv")
for line in f:
    div = line.split(',')
    labels.append(div[0])
    PYTORCH.append(float(div[1]))
    AUTOTVM.append(float(div[2]))
    ANSOR.append(float(div[3]))
    FAMILYSEER.append(float(div[4]))
# print('======== loaded ========')
# print(redundancy)

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

# brewer2mpl.get_map args: set name set type number of colors
# bmap = brewer2mpl.get_map('RdYlBu', 'diverging', 5)

mycolors = [ '#3399CC', '#CCCCCC', '#f7fcb9','#addd8e' ]
#mycolors = [ '#fc8d59', '#99d594' ]

# # set color
#plt.rcParams['axes.color_cycle'] = mycolors
# print(plt.style.available)
# plt.style.use('seaborn-darkgrid')

# Plot configuration
f, ax1 = plt.subplots(1, 1, figsize=(12, 2.5))
plt.xticks(rotation=00)
#plt.title('Latency Comparison on CP',fontsize='large',fontweight='bold') 
# make the figure tight
ax1.set_xlim(-1,len(ANSOR)*2-1)
xlabels = [i for i in range(0,len(ANSOR)*2,2)]
ax1.set_xticks(xlabels)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.0,1.1)
plt.tick_params(labelsize=12)
plt.rcParams['font.family'] = "Times New Roman"

# ax1.set_xlabel('Cases')
ax1.set_ylabel('Speedup',fontsize=14)
font = {'style':'normal','weight':'bold'}

#x = [ i for i in range(1, len(labels))]
#x = [i * 3 for i in x]

plt.fill_between([1,3],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([5,7],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([9,11],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([13,15],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([17,19],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([21,23],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([25,27],0.0,3,facecolor='grey',alpha=0.2)
plt.fill_between([29,31],0.0,3,facecolor='grey',alpha=0.2)

#x1= x
width = 0.4
bar0 = ax1.bar([i-width*3/2 for i in xlabels], PYTORCH, color=mycolors[3], width=width, label='XLA', edgecolor='black', linewidth=0.1,zorder=1,alpha=0.6)
bar1 = ax1.bar([i-width*1/2 for i in xlabels], AUTOTVM, color=mycolors[2], width=width, label='AutoTVM', edgecolor='black', linewidth=0.1,zorder=1,alpha=0.6)
bar2 = ax1.bar([i+width*1/2 for i in xlabels], ANSOR, color=mycolors[1], width=width, label='Ansor', edgecolor='black', linewidth=0.1,zorder=1)
bar3 = ax1.bar([i+width*3/2 for i in xlabels], FAMILYSEER, color=mycolors[0], width=width, label='FamilySeer', edgecolor='black', linewidth=0.1,zorder=5)
#bar4 = ax1.bar([i+3*width/2 for i in xlabels], FAMILYSEER_GPU_PARALLEL, color=mycolors[0], width=width, label='FamilySeer+GPU+PARALLEL', edgecolor='black', linewidth=0.1,zorder=5)
add_value_labels(ax1)
#ax1.annotate(s='',xy=(-0.5,0),xytext=(-0.5,-32),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
offset=-1
#X轴竖线
#ax1.annotate(s='',xy=(offset,0.0),xytext=(offset,-0.55),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"ResNet50_v1",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"ResNet152_v2",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"Mobilenet",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"Mobilenetv2",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"ViT-H*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"BERT-L*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"RoBERTa-L*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")

offset=offset+4
ax1.annotate(text='',xy=(offset,0.0),xytext=(offset,-0.25),arrowprops=dict(facecolor='black', headlength=20, width=0.1, headwidth=0.01))
ax1.text(offset+2,-0.25,"GPT2-S*",fontsize=12,fontdict=font,verticalalignment="bottom",horizontalalignment="center")


legend1=ax1.legend(loc='upper right', ncol=4,edgecolor='Black',facecolor='white',fontsize=12,framealpha=0.2, bbox_to_anchor=(1,1.15),borderaxespad = 0.)
#legend2=ax1.legend(loc='upper right', ncol=4,title="GPU",edgecolor='Black',facecolor='grey',framealpha=0.2)

f.savefig('figure-7.pdf', bbox_inches = 'tight')
#plt.show()
