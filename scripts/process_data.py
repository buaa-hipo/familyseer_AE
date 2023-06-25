import tarfile


def get_number(improve_ratio,target_num):
    # -*- coding: utf-8 -*-
    #import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import csv
    import math
    from mpl_toolkits.axes_grid1 import axes_grid

    pretrain_cpu =[691,4462,827,2634,1021,1595,663,2784]

    pretrain_gpu =[1141,1629,3356,2492,2873,3578+161,950,5500]
    GPU_offset =[1.11,1.1,1.06,1.11,1.097,1,1.11,1]

    #GPU
    models = [ 'resnet50_v1', 
            'resnet152_v2',
            'mobilenet0.5', 
            'mobilenetv2_0.5', 
            'vit_huge',
            'bert_large',
            'roberta_large',
            'gpt2']


    methods = ['family', 'default']

    targets = ['silver_4210', 'NVIDIA_V100']

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
    data_gpu_family = [get_x_y_list(get_filename(i, methods[0], targets[target_num])) for i in models]
    data_gpu_default = [get_x_y_list(get_filename(i, methods[1], targets[target_num])) for i in models]

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
        #print (default_max,"....",default_min)
        gpu_default_offset.append((default_max-default_min)*percent)
        gpu_default_min.append(default_min)
        gpu_default_max.append(default_max)
        #for j in range(len(data_gpu_family[i][1])):
        #    data_gpu_family[i][1][j] += pretrain_cpu[i]
            #data_gpu_family[i][0][j] = data_gpu_family[i][0][j]*GPU_offset[i]

    #性能达标位置


    #截取高收益段
    for i in range(len(models)):
        high_profit = gpu_default_min[i]+gpu_default_offset[i]
        high_profit = high_profit+(gpu_default_max[i]-high_profit)*improve_ratio
        #print(high_profit)
        for index,k in enumerate(data_gpu_default[i][0]):
            if high_profit >= k:
                data_gpu_default[i][0]=data_gpu_default[i][0][:index]
                data_gpu_default[i][1]=data_gpu_default[i][1][:index]
                break
        print(models[i])
        print (data_gpu_default[i])
        
    #对曲线进行抚平
    for i in range(len(models)):
        default_min = 99999.9
        family_min = 99999.9
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

    #偏差数据构建：基于default
    data_offset = []
    for i in range(len(models)):
        #对齐数据
        data_gpu_family[i][0][0]=data_gpu_default[i][0][0]
        data_gpu_family[i][1][0]=data_gpu_default[i][1][0]
        #计算差距
        data = [[],[]]
        for j in range(1,len(data_gpu_default[i][0])):
            default_time = data_gpu_default[i][1][j]
            default_latency = data_gpu_default[i][0][j]
            offset_time = 0
            #print(default_latency)
            flag = False
            for index, k in enumerate(data_gpu_family[i][0]):
                #print (k)
                #input()
                if default_latency >= k:
                    #print("in")
                    ratio = (data_gpu_family[i][0][index]-data_gpu_family[i][0][index-1])/(data_gpu_family[i][1][index]-data_gpu_family[i][1][index-1])
                    offset = default_latency - k
                    family_time = data_gpu_family[i][1][index]+ offset * ratio
                    offset_time = default_time-family_time
                    #data_offset[i].append(default_time-family_time)
                    break
            data[0].append(default_latency)
            data[1].append(offset_time)
        data_offset.append(data)
    print("===================================")
    #print(data_offset)

    #print("Time saved at {}%".format((1-improve_ratio)*100))
    for i,model in enumerate(models):
        #for j in range(len(data_gpu_default[i][0])):
            #if data_gpu_default[i][0][j] == data_offset[i][0][-1]:
            #print(model,",",data_gpu_default[i][0][-1],",",data_gpu_default[i][1][-1],model,",",data_offset[i][0][-1],",",data_offset[i][1][-1])
            ansor = data_gpu_default[i][1][-1]
            familyseer = data_gpu_default[i][1][-1] - data_offset[i][1][-1]
            if target_num == 0:
                if autotvm_cpu[int(improve_ratio*10)][i] != 0:
                    autotvm = autotvm_cpu[int(improve_ratio*10)][i]
                    speedup_cpu.append("{:.0f}%,{},{},{}".format((1-improve_ratio)*100,ansor/ansor,ansor/familyseer,ansor/autotvm))
                else:
                    speedup_cpu.append("{:.0f}%,{},{},{}".format((1-improve_ratio)*100,ansor/ansor,ansor/familyseer,0))
            if target_num == 1:
                if autotvm_gpu[int(improve_ratio*10)][i] != 0:
                    autotvm = autotvm_gpu[int(improve_ratio*10)][i]
                    speedup_gpu.append("{:.0f}%,{},{},{},{},{}".format((1-improve_ratio)*100,ansor/ansor,ansor/familyseer,ansor/familyseer*xgb[int(improve_ratio*10)][i],ansor/familyseer*parallel[int(improve_ratio*10)][i],ansor/autotvm))
                else:
                    speedup_gpu.append("{:.0f}%,{},{},{},{},{}".format((1-improve_ratio)*100,ansor/ansor,ansor/familyseer,ansor/familyseer*xgb[int(improve_ratio*10)][i],ansor/familyseer*parallel[int(improve_ratio*10)][i],0))    
            #print()
            #   break
    #for i,model in enumerate(models):
        #for j in range(len(data_offset[i][0])):
        #    if data_offset[i][0][-1] == data_offset[i][0][j]:
            #print()
            #   break
    if improve_ratio == 0.0:
        #print("===================================")
        #print("E2E performance")
        for i,model in enumerate(models):
            #print(data_gpu_default[i][0][-1],",",data_gpu_family[i][0][-1])
            ansor = data_gpu_default[i][0][-1]
            familyseer = data_gpu_family[i][0][-1]
            autotvm = speed_autotvm[target_num][i]
            xla = speed_xla[target_num][i]
            
            
            if target_num == 0:
                if autotvm == 0:
                    latency_cpu.append("{},{},{},{},{}".format('CPU',ansor/xla,0,ansor/ansor,ansor/familyseer,0))
                elif xla ==0 :
                    latency_cpu.append("{},{},{},{},{}".format('CPU',0,ansor/autotvm,ansor/ansor,ansor/familyseer,0))
                else:
                    latency_cpu.append("{},{},{},{},{}".format('CPU',ansor/xla,ansor/autotvm,ansor/ansor,ansor/familyseer,0))
            else:
                if autotvm == 0:
                    latency_gpu.append("{},{},{},{},{}".format('GPU',ansor/xla,0,ansor/ansor,ansor/familyseer,0))
                elif xla ==0 :
                    latency_gpu.append("{},{},{},{},{}".format('GPU',0,ansor/autotvm,ansor/ansor,ansor/familyseer,0))
                else:
                    latency_gpu.append("{},{},{},{},{}".format('GPU',ansor/xla,ansor/autotvm,ansor/ansor,ansor/familyseer,0))
            
latency_cpu = []
latency_gpu = []
speedup_cpu = []
speedup_gpu = []
improve_ratio = [0.2,0.1,0.0]
autotvm_cpu= [
    [0,0,0,9301.79839,0,0,0,0],
    [6963.470182,0,0,9005.28039,0,0,0,0],
    [5034.14,9196.780743,0,8597.56749,0,0,0,0]
    ]
autotvm_gpu= [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [26130.31,25648.15,0,34957.01547,0,0,0,0]
    ]

speed_autotvm = [
    [
        13.75,
        41.44,
        1.56,
        1.78,
        839.98,
        410.32,
        400.02,
        16369.09

    ],
    [
        2.03,
        6.1,
        0.38,
        0.45,
        787.53,
        29.34,
        35.12,
        0
    ]
]
speed_xla = [
    [
        86.2,
        183.4,
        41.7,
        49.5,
        3510.7,
        690.59,
        745.55,
        82853.44
    ],
    [
        45.6,
        58.1,
        37.9628,
        43.6655,
        213.4,
        21.8485,
        25.1542,
        36001.01
    ]
]

xgb = [
    [
        1.030638161,
        1.030638161,
        1.041949953,
        1.095782415,
        1.025145366,
        1.024308184,
        1.01848352,
        1.010907475
        ],
    [
        1.024627303,
        1.039865214,
        1.033703565,
        1.075640437,
        1.020359142,
        1.019364752,
        1.014713693,
        1.008635461
        ],
    [
        1.019283808,
        1.031442478,
        1.026119028,
        1.058882851,
        1.015692091,
        1.015549337,
        1.011361844,
        1.006264257
        ]
]
parallel = [
    [
        1.62468635,
        1.62468635,
        1.624137865,
        1.785805719,
        1.593858898,
        1.4974984,
        1.562165941,
        1.727447838
    ],
    [
        1.524118367,
        1.481572725,
        1.525292701,
        1.63980195,
        1.50273882,
        1.431062881,
        1.484107669,
        1.614595776
     ],
    [
        1.437824079,
        1.400241075,
        1.435458157,
        1.517558163,
        1.420588672,
        1.377247504,
        1.411998813,
        1.508385255
    ]
]

for target_num in range(0,2):
    '''
    if target_num ==0:
        print("CPU Summary:")
    else:
        print("\n")
        print("GPU Summary:")
    '''
    for ratio in improve_ratio:
        get_number(ratio,target_num)
#print(speedup_cpu)
fcpu = open('speedup_cpu.csv','w+')
for i in range(0,8):
    for j in range(i,len(speedup_cpu),8):
        print(speedup_cpu[j],file=fcpu)
        fcpu.flush()
fcpu.close()

fgpu = open('speedup_gpu.csv','w+')
for i in range(0,8):
    for j in range(i,len(speedup_gpu),8):
        print(speedup_gpu[j],file=fgpu)
        fgpu.flush()
fgpu.close()

#print(latency_cpu)
flatency = open('speedup_latency_n.csv','w+')
for i in range(0,8):
    print(latency_cpu[i],file=flatency)
    print(latency_gpu[i],file=flatency)
    flatency.flush()
flatency.close()

