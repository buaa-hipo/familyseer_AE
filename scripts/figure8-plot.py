# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
from mpl_toolkits.axes_grid1 import axes_grid
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=1)
args = parser.parse_args()
models = ['gpt2']

methods = ['family', 'default']

targets = ['NVIDIA_V100']

def get_filename(model, method, target):
    return "./log_data/realbench_log/{}_{}_{}_B1.step".format(model, target, method)

def get_x_y_list(filename):
    data = [[], []]
    f = open(filename, 'r')

    symbol = dict()
    symbol['latency'] = ['Mean inference time (std dev): ', ' ms']
    symbol['time'] = ['ms) ', ' s']

    for line in f:
        if line.find(symbol['latency'][0]) != -1 and line.find(symbol['time'][0]) != -1:
            strings = symbol['latency'][0]
            first = line.find(strings) + len(strings)
            strings = symbol['latency'][1]
            end = line.find(strings)
            temp = line[first:end]
            try:
                temp = float(temp)
            except:
                temp = None
            data[0].append(temp)    
            
            line = line[end+len(strings):]
            strings = symbol['time'][0]
            first = line.find(strings) + len(strings)
            strings = symbol['time'][1]
            end = line.find(strings)
            temp = line[first:end]
            temp = int(temp)
            #temp = None if temp == 0 else temp
            data[1].append(temp)
    f.close()

    return data

##################################

#if __name__ == '__main__':
#filename = get_filename(models[0], methods[0], targets[0]) 
#data = get_x_y_list(filename)
#print(data)

#准备数据
data_gpu_family = [get_x_y_list(get_filename(i, methods[0], targets[0])) for i in models]
data_gpu_default = [get_x_y_list(get_filename(i, methods[1], targets[0])) for i in models]



#获取提升幅度
gpu_default_offset = []
gpu_default_min = []
gpu_default_max = []
percent = 0.05
for i in range(len(models)):
    default_max = 0.0
    default_min = 99999.0
    for k in data_gpu_default[i][0]:
        if default_max < k:
            default_max = k
        if default_min > k:
            default_min = k
    print (default_max,"....",default_min)
    gpu_default_offset.append((default_max-default_min)*percent)
    gpu_default_min.append(default_min)
    gpu_default_max.append(default_max)
    #for j in range(len(data_gpu_family[i][1])):
    #    data_gpu_family[i][1][j] += pretrain_cpu[i]
        #data_gpu_family[i][0][j] = data_gpu_family[i][0][j]*GPU_offset[i]

'''
#对曲线进行抚平
for i in range(len(models)):
    default_min = 99999
    family_min = 99999
    for k in range(len(data_gpu_default[i][1])):
        if (data_gpu_default[i][0][k]<default_min):
            default_min = data_gpu_default[i][0][k]
            #break
        else:
            data_gpu_default[i][0][k] = default_min
    
    for j in range(len(data_gpu_family[i][1])):
        if (data_gpu_family[i][0][j]<family_min):
            family_min = data_gpu_family[i][0][j]
            #break
        else:
           data_gpu_family[i][0][j] = family_min
'''

#截取高收益段
for i in range(len(models)):
    high_profit = gpu_default_min[i]+gpu_default_offset[i]
    for index,k in enumerate(data_gpu_default[i][0]):
        if high_profit >= k:
            data_gpu_default[i][0]=data_gpu_default[i][0][:index]
            data_gpu_default[i][1]=data_gpu_default[i][1][:index]
    #print (data_gpu_default[i])


#给familyseer增加pretrain的时间

for i in range(len(models)):
    for k in range(len(data_gpu_default[i][1])):
        if data_gpu_default[i][0][k]: #>0, not None
            break
    
    #for j in range(len(data_gpu_family[i][1])):
    #    data_gpu_family[i][1][j] += pretrain_cpu[i]
        #data_gpu_family[i][0][j] = data_gpu_family[i][0][j]*GPU_offset[i]


#开始画图
plt.rc('font',family='Times New Roman')
plt.rcParams['figure.figsize'] = (4.0, 2.5) 
#plt.rcParams['figure.figsize'] = (17.5, 6.0)
nrow=1
ncol=1
fig, ax = plt.subplots()

def draw_picture(data_family, data_default):
    data_fs=14
    fs = 12
    fs_ylabel=14
    padding = 0.4
    for i in range(1*1):
        col = i % ncol
        row = math.floor(i / ncol)
        l1, = ax.plot(data_family[i][1],data_family[i][0],linestyle='--')
        l2, = ax.plot(data_default[i][1],data_default[i][0],alpha=0.7, marker='o',markersize=2)

        yvalues = [v for v in data_default[i][0]+data_family[i][0] if v is not None]
        print("model: {}, max: {}, min: {}".format(models[i], max(yvalues), min(yvalues)))
        l_max = max(yvalues)
        l_min = min(yvalues)
        yticks = []
        #for jj in range(11):
        #    yticks.append(l_min + jj/10.0*(l_max-l_min))

        xlabels = [i * 2000 for i in range(6)]
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=fs)
        ylabels = [300 + i * 30 for i in range(6)]
        ax.set_yticks(ylabels)
        ax.set_yticklabels(ylabels, fontsize=fs)

        #ax.set_yticks(yticks)
        #ax[row][col].set_yticklabels(strong_ylabel_list, fontsize=fs)
        #ax[row][col].set_xticks()
        ax.set_xlim([0, data_default[i][1][-1]])
        #print(strong_log_yvalue_list)
        ax.legend((l1,l2), ('FamilySeer', 'Ansor'), fontsize=12, loc='upper right')
        #ax.title.set_text("{}".format(models_name[i]))
        #ax[row][col].title.set_position([0.5, -5])
    ax.set_ylabel('Inference Latency (ms)',fontsize=fs_ylabel)
    #ax[1][0].set_ylabel('Inference Latency (ms)',fontsize=fs_ylabel)
    ax.set_xlabel('Used Time (s)',fontsize=fs_ylabel)
    #ax[0][1].set_xlabel('Used Time (s)',fontsize=fs_ylabel)
    #ax[0][2].set_xlabel('Used Time (s)',fontsize=fs_ylabel)

    #ax[1][0].set_xlabel('Used Time (s)',fontsize=fs_ylabel)
    #ax[1][1].set_xlabel('Used Time (s)',fontsize=fs_ylabel)
    #ax[1][2].set_xlabel('Used Time (s)',fontsize=fs_ylabel)

    fig.subplots_adjust(hspace = 0.1)
 
####################

draw_picture(data_gpu_family, data_gpu_default)
plt.tight_layout()
plt.show()

fig.savefig('./figure-8.pdf')





