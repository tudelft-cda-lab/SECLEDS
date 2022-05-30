import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math

        
        
        
def theoretical():        
    ## Distance computations over growing k
    _k = [2, 3, 5, 10, 15, 20, 30, 50]
    _p = [1, 3, 5, 10]
    d = 1
    b = 100
    n = 10000#range(200, 100000, 100)
    dimension = 1000
    pal_b = sns.color_palette("dark", len(_k))
    pal_s = sns.color_palette("tab20b", len(_k))
    linestyle = ['--', '-.', '-', ':']
    vert = [100000, 20000, 4000, 800]

    params = {'mathtext.default': 'regular', 'font.size': 16 } 
             
    plt.rcParams.update(params)
    ### Time complexity
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    axs[0].set_title('Scaling with k clusters, n=10000')
    axs[0].set(xlabel='Number of clusters (k)', ylabel=r'$log_{10}(no. distance\ computations)$')



    maxi , mini = 0 ,0
    bandit = [math.log(k*(4*n + (n*min(2*n, math.log(n, 2) + b))),10) for k in _k] #k*(4n + n*min[2n, logn + b])

    for pid, p in enumerate(_p):
        comp_b, comp_s = [0]*len(_k), [0]*len(_k)
        seqclu = [math.log((n*k*p) + (k*b), 10) for k in _k] #n*((d*k*p)+(2*p))
        axs[0].plot(_k, seqclu, label='SECLEDS [p='+str(p)+']', color=pal_s[3-pid], linestyle='-', marker='o')
    axs[0].plot(_k, bandit, label='BanditPAM', color='red', linestyle=':', marker='s')
    plt.xticks(_k, _k)


    ## Distance computations over growing n
    k = 5#[2, 3, 5, 10, 15, 20, 30, 50]
    _p = [1, 3, 5, 10]
    d = 1
    b = 100
    _n = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]#[10 ** exponent for exponent in range(2,7)]
    dimension = 1000
    pal_b = sns.color_palette("dark", len(_n))
    pal_s = sns.color_palette("tab20b", len(_n))
    linestyle = ['--', '-.', '-', ':']
    vert = [100000, 20000, 4000, 800]

    ### Time complexity
    axs[1].set_title('Scaling with stream size n, k=5')
    axs[1].set(xlabel='Stream size (n)', ylabel=r'$log_{10}(no. distance\ computations)$')


    maxi , mini = 0 ,0
    bandit = [math.log(k*(4*n + (n*min(2*n, math.log(n, 2) + b))),10) for n in _n] #k*(4n + n*min[2n, logn + b])

    for pid, p in enumerate(_p):
        seqclu = [math.log((n*k*p) + (k*b), 10) for n in _n] #n*((d*k*p)+(2*p))
        axs[1].plot(_n, seqclu, label='OPSECLU [p='+str(p)+']', color=pal_s[3-pid], linestyle='-', marker='o', markersize=8, linewidth=2)
    axs[1].plot(_n, bandit, label='BanditPAM', color='red', linestyle=':', marker='s')
    #ax = plt.gca()	
    plt.xticks(_n, _n, rotation='45')
    axs[0].set_xticks(ticks=_k, labels=_k)
    axs[1].set_xticks(ticks=_n, labels=_n, minor=True, rotation=45)

    axs[1].tick_params(rotation=45)

    axs[0].grid(True, which='both')
    axs[1].grid(True, which='both')
    h, l = [(a) for a in axs[0].get_legend_handles_labels()]

    fig.legend( h, l,
               loc="lower center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               ncol=5
               )
    fig.subplots_adjust(bottom=0.21, wspace=0.18, top=0.74) # or whatever
    plt.show()

theoretical()



data = []

pal_s = ['maroon', 'red',
'blue', 'dodgerblue', 'steelblue',
'darkgreen']#sns.color_palette("pastel", 6)
pal_b = ['salmon', 'red',
'blue', 'dodgerblue', 'steelblue',
'darkgreen']#sns.color_palette("pastel", 6)
pal_p = sns.color_palette("tab20b", 4)
pal_p = list(reversed(list(pal_p)))



linestyle = ['-', '-', '-', '-', '-', ':']#, '--', ':']
markers = ['o', 'o', 'o', 'o', 'o', 's'] #['*', 'o', '1', '*', 'x', 'd']

blobx = [5000, 10000, 15000, 20000, 25000, 30000, 50000]
sinex = [2000, 5000, 10000, 15000, 20000, 30000]


### Blobs
blobf1 = {
'SECLEDS-rand':
[0.6551,
0.6886 ,
0.6759 ,
0.6704 ,
0.67097,
0.6892 ,
0.67656],
'SECLEDS':
[0.9108,
0.89406,
0.8875 ,
0.8962 ,
0.8877 ,
0.8998 ,
0.8855 ,

],
'MB K-means':
[0.895,
0.888 ,
0.888 ,
0.9144,
0.9071,
0.9199,
0.8832,

],
'CluStream':
[0.2784,
0.2927 ,
0.3069 ,
0.3217 ,
0.2752 ,
0.2739 ,
0.297  ,

],
'StreamKM++':
[0.6105,
0.5988 ,
0.6037 ,
0.568  ,
0.517  ,
0.5388 ,
0.535  ,

],
'BanditPAM':
[0.9546,
0.9533 ,
0.9534 ,
0.9523 ,
0.9521 ,
0.952  ,
0.949  ,

],

}

blobttc = {
'SECLEDS-rand':
[66.66,
139.86,
215.96,
286.97,
339.81,
402.39,
670.96,
],
'SECLEDS':
[68.60,
132.84,
204.62,
283.97,
359.23,
403.62,
686.17,

],
'MB K-means':
[7.62,
14.61,
22.44,
29.78,
40.43,
51.44,
76.22,

],
'CluStream':
[47.79,
82.39 ,
128.14,
177.04,
221.43,
284.09,
422.07,

],
'StreamKM++':
[3.4,
5.76,
9.90,
12.3,
14.4,
19.2,
29.8,

],
'BanditPAM':
[36.83,
105.05,
313.34,
373.53,
695.45,
903.45,
1842.2,

],
}

### Errors
blobf1_err = {
'SECLEDS-rand':
[0.0166,
0.0160,
0.0144,
0.0175,
0.0207,
0.019 ,
0.0172,
],
'SECLEDS':
[
0.0089,
0.0083,
0.0102,
0.0161,
0.0195,
0.0161,
0.0042,
],
'MB K-means':
[0.0159,
0.0102 ,
0.0099 ,
0.0106 ,
0.0144 ,
0.0145 ,
0.015  ,

],
'CluStream':
[0.0167,
0.0168 ,
0.0125 ,
0.0212 ,
0.0141 ,
0.0095 ,
0.0133 ,

],
'StreamKM++':
[
0.0095,
0.0083,
0.0102,
0.0089,
0.0067,
0.0096,
0.0190,
],
'BanditPAM':
[0,
0 ,
0 ,
0 ,
0 ,
0 ,
0

],

}

blobttc_err = {
'SECLEDS-rand':
[0.3477,
7.966  ,
14.038 ,
13.364 ,
8.439  ,
2.790  ,
2.410  ,
],
'SECLEDS':
[2.323,
0.687 ,
4.374 ,
12.247,
24.480,
3.405 ,
7.124 ,

],
'MB K-means':
[
0.617,
0.027,
0.472,
0.517,
3.351,
2.710,
0.654,

],
'CluStream':
[
5.967,
0.075,
1.358,
9.535,
14.13,
31.28,
1.147,
],
'StreamKM++':
[
0.504,
0.020,
0.795,
0.747,
0.135,
0.841,
0.238,
],
'BanditPAM':
[
6.589,
9.966,
19.72,
15.43,
25.40,
10.61,
29.55,


],

}

## Sine drift = off
sinef1 = {
'SECLEDS-dtw':
[1   ,
0.999,
0.999,
1    ,
1    ,
0.999,
],
'SECLEDS':
[0.9996,
0.9999 ,
0.9337 ,
0.999  ,
1      ,
1      ,

],
'MB K-means':
[0.8666,
0.9001 ,
1      ,
0.9333 ,
0.8666 ,
1      ,

],
'CluStream':
[0.41,
0.409,
0.408,
0.408,
0.409,
0.406,

],
'StreamKM++':
[0.7937,
0.8287 ,
0.8417 ,
0.7224 ,
0.6864 ,
0.66311,

],
'BanditPAM':
[1,
1 ,
1 ,
1 ,
1 ,
1 ,

],

}
sinettc = {
'SECLEDS-dtw':
[24.27,
58.97 ,
117.5 ,
185.15,
240.38,
353.94,
],
'SECLEDS':
[9.17,
22.01,
44.49,
65.90,
87.53,
130.9,

],
'MB K-means':
[2.83,
6.691,
13.51,
21.09,
27.29,
41.25,

],
'CluStream':
[18.84,
43.526,
91.906,
133.93,
174.74,
269.23,

],
'StreamKM++':
[2.36,
5.475,
11.75,
17.00,
22.25,
34.45,

],
'BanditPAM':
[1.799,
6.88  ,
20.966,
43.654,
68.69 ,
139.98,

],

}

## Errors

sinef1_err = {
'SECLEDS-dtw':
[0,
0 ,
0 ,
0 ,
0 ,
0 ,

],
'SECLEDS':
[0.00028,
0       ,
0.059   ,
0       ,
0       ,
0       ,


],
'MB K-means':
[0.0516,
0.048  ,
0      ,
0.059  ,
0.073  ,
0      ,


],
'CluStream':
[0.00025,
0.0006  ,
0.0006  ,
0.00127 ,
0.0015  ,
0.0013  ,


],
'StreamKM++':
[0.0581,
0.0368 ,
0.0979 ,
0.0723 ,
0.064  ,
0.0566 ,


],
'BanditPAM':
[0,
0 ,
0 ,
0 ,
0 ,
0 ,

],

}

sinettc_err = {
'SECLEDS-dtw':
[0.08,
0.477,
0.948,
8.61 ,
6.905,
3.301,

],
'SECLEDS':
[0.151,
0.0548,
0.361 ,
0.524 ,
0.1711,
0.509 ,


],
'MB K-means':
[
0.107,
0.004,
0.129,
0.291,
0.116,
0.497,

],
'CluStream':
[0.213,
0.1003,
0.765 ,
1.189 ,
0.627 ,
1.402 ,


],
'StreamKM++':
[0.003,
0.0061,
0.0857,
0.178 ,
0.047 ,
0.156 ,


],
'BanditPAM':
[0.170,
0.6534,
0.75  ,
7.809 ,
8.289 ,
5.04  ,

],

}

## Sine drift = 0n
sinef1_drift = {
'SECLEDS-dtw':
[1,
1,
0.9941,
1,
1,
1,
],
'SECLEDS':
[0.9665,
0.9997 ,
1      ,
1      ,
1      ,
1      ,

],
'MB K-means':
[0.4468,
0.43499,
0.4413 ,
0.4607 ,
0.495  ,
0.47298,

],
'CluStream':
[0.3811,
0.3881 ,
0.3869 ,
0.3785 ,
0.3819 ,
0.37338,

],
'StreamKM++':
[0.53884,
0.40708 ,
0.3825  ,
0.4114  ,
0.3645  ,
0.3507  ,

],
'BanditPAM':
[0.455,
0.4291,
0.4171,
0.4088,
0.435 ,
0.4272,

],

}
sinettc_drift = {
'SECLEDS-dtw':
[24.782,
62.229 ,
125.259,
199.825,
251.051,
377.05 ,
],
'SECLEDS':
[8.66,
21.89,
44.34,
65.41,
88.61,
132.5,

],
'MB K-means':
[2.6108,
6.566  ,
13.3291,
19.774 ,
27.782 ,
40.411 ,

],
'CluStream':
[35.113,
90.6516,
188.775,
266.623,
368.707,
552.129,

],
'StreamKM++':
[4.5644,
13.0562,
26.925 ,
35.944 ,
49.046 ,
73.319 ,

],
'BanditPAM':
[1.2725,
6.5906 ,
54.3349,
71.5   ,
114.392,
226.29 ,

],

}

## Errors

sinef1_drift_err = {
'SECLEDS-dtw':
[0,
0,
0.00425,
0,
0,
0

],
'SECLEDS':
[
0.031,
0.000161,
0,
0,
0,
0,

],
'MB K-means':
[
0.0131,
0.0088,
0.014 ,
0.0245,
0.0253,
0.0173,

],
'CluStream':
[
0.0021,
0.0022,
0.0029,
0.0038,
0.0036,
0.0032,

],
'StreamKM++':
[
0.0504,
0.015 ,
0.01  ,
0.028 ,
0.014 ,
0.0063,

],
'BanditPAM':
[0,
0 ,
0 ,
0 ,
0 ,
0 ,

],

}

sinettc_drift_err = {
'SECLEDS-dtw':
[0.062,
0.129 ,
0.144 ,
6.962 ,
0.4538,
3.1456,

],
'SECLEDS':
[
0.013,
0.027,
0.055,
0.176,
0.158,
0.65 ,

],
'MB K-means':
[
0.0037,
0.0054,
0.0514,
0.047 ,
0.062 ,
0.321 ,

],
'CluStream':
[
0.2355,
0.183 ,
3.097 ,
0.678 ,
6.946 ,
14.029,

],
'StreamKM++':
[
0.007,
0.881,
1.729,
0.051,
0.037,
0.223,

],
'BanditPAM':
[0.12,
0.625,
0.71 ,
7.79 ,
8.463,
10.48,

],

}

## Prototypes scaling

scalewithpf1 = {
'SECLEDS [p=1]' :
[0.2873,
0.27064,
0.256  ,
0.254  ,
0.2535 ,
0.2517 ,

],

'SECLEDS [p=3]' :
[0.8469,
0.9372 ,
0.8552 ,
0.9566 ,
0.9553 ,
0.9277 ,

],
'SECLEDS [p=5]' :
[0.906,
0.9121,
0.9386,
0.9952,
0.9411,
0.9528,

],
'SECLEDS [p=10]' :
[0.991,
0.9983,
0.9456,
0.9988,
0.9461,
0.9989,

],

}

scalewithpttc = {
'SECLEDS [p=1]' :
[2.94,
7.265,
14.72,
21.98,
29.24,
45.35,

],

'SECLEDS [p=3]' :
[7.488,
18.529,
37.197,
55.59 ,
74.167,
112.52,

],
'SECLEDS [p=5]' :
[11.49,
28.363,
57.193,
85.81 ,
114.23,
172.92,

],
'SECLEDS [p=10]' :
[21.93,
53.913,
108.81,
162.43,
217.52,
327.88,

],

}

## Errors

scalewithpf1_err = {
'SECLEDS [p=1]' :
[0.0044,
0.0042 ,
0.001  ,
0.00106,
0.00064,
0.00034,

],

'SECLEDS [p=3]' :
[0.043,
0.0075,
0.059 ,
0.0027,
0.0068,
0.029 ,

],
'SECLEDS [p=5]' :
[0.0396,
0.0389 ,
0.049  ,
0.001  ,
0.0457 ,
0.029  ,

],
'SECLEDS [p=10]' :
[0.0038,
0.0011 ,
0.0466 ,
0.00089,
0.0468 ,
0.00034,

],

}

scalewithpttc_err = {
'SECLEDS [p=1]' :
[0.0094,
0.006  ,
0.14   ,
0.049  ,
0.06   ,
0.5833 ,

],

'SECLEDS [p=3]' :
[0.0352,
0.057  ,
0.318  ,
0.212  ,
0.188  ,
0.741  ,

],
'SECLEDS [p=5]' :
[0.025,
0.0511,
0.587 ,
0.271 ,
0.2944,
1.09  ,

],
'SECLEDS [p=10]' :
[0.018,
0.0935,
1.158 ,
0.1886,
0.5833,
2.272 ,

],

}


import numpy as np
yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
def plotit_p():
    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    plt.suptitle('Sine-curve, k=4')
    rec = []
    for pid, (xid,obj, obj_err, ylab, xticks, pale) in enumerate([
    (0,scalewithpf1, scalewithpf1_err, 'F1', sinex, pal_p), 
    (1,scalewithpttc, scalewithpttc_err, 'Runtime (sec)', sinex, pal_p), 
    ]):
        axs[xid].grid(True, which='both')
        axs[xid].set(xlabel='Stream size (n)', ylabel=ylab)
        axs[xid].set_xticks(ticks=xticks, labels=xticks, minor=True, rotation=45)
        axs[xid].tick_params(rotation=45)

            
        for idx,(config,vals) in enumerate(obj.items()):
            if pid == 2:
                if idx == 0:
                    axs[xid].errorbar(xticks, vals, yerr=obj_err[config], label=config, color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2)
                else:
                    axs[xid].errorbar(xticks, vals, yerr=obj_err[config], label='_nolegend_', color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2)
            else:
                axs[xid].errorbar(xticks, vals, yerr=obj_err[config], label=config, color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2)


    h, l = [(a) for a in axs[1].get_legend_handles_labels()]

    fig.legend( h, l,
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           ncol=4
           )
    fig.subplots_adjust(bottom=0.21, wspace=0.18, top=0.74)#bottom=0.2, top=0.92) # or whatever
    


    plt.show()
    #plt.savefig(name+'.png')

    



# Plot major results
def plotit_configs():#
    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(2, 3, figsize=(8,6))
    axs[0, 0].set_title('Blobs, k=10, p=5')
    axs[0, 1].set_title('Sine-curve, k=3, p=5')
    axs[0, 2].set_title('Sine-curve-drifted, k=3, p=5')

    rec = []
    for pid, (xid,yid,obj, obj_err, ylab, xticks, pale) in enumerate([
    (0,0,blobf1, blobf1_err, 'F1', blobx, pal_b), 
    (1,0,blobttc, blobttc_err, 'Runtime (sec)', blobx, pal_b), 
    (0,1,sinef1, sinef1_err, 'F1', sinex, pal_s), 
    (1,1,sinettc, sinettc_err, 'Runtime (sec)',sinex, pal_s),
    (0,2,sinef1_drift, sinef1_drift_err, 'F1', sinex, pal_s), 
    (1,2,sinettc_drift, sinettc_drift_err, 'Runtime (sec)',sinex, pal_s)]):

        axs[xid,yid].grid(True, which='both')


        if xid == 1 and yid in [0, 1, 2]:
            axs[xid,yid].set(xlabel='Stream size (n)')
            axs[xid,yid].xaxis.set_ticks_position('bottom')
            axs[xid,yid].set_xticks(ticks=xticks, labels=xticks, minor=True, rotation=45)
            axs[xid,yid].tick_params(rotation=45)
            
        if (xid == 0 and yid == 0) or (xid == 1 and yid == 0):
            axs[xid,yid].set(ylabel=ylab)
        if xid == 0 and yid in [0, 1, 2]:    
            axs[xid,yid].xaxis.set_ticklabels([])
            axs[xid,yid].set_xticks(ticks=xticks, labels=[], minor=True, rotation=45)
        if 'F1' in ylab:
            axs[xid,yid].set_ylim([0,1.1])
            
        for idx,(config,vals) in enumerate(obj.items()):
           
            if pid == 2:
                if idx == 0:
                    axs[xid,yid].errorbar(xticks, vals, yerr=obj_err[config], label=config, color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2, alpha=0.8)
                else:
                    axs[xid,yid].errorbar(xticks, vals, yerr=obj_err[config], label='_nolegend_', color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2, alpha=0.8)
            else:
                axs[xid,yid].errorbar(xticks, vals, yerr=obj_err[config], label=config, color=pale[idx], linestyle=linestyle[idx], marker=markers[idx],  elinewidth=2, markersize=8, linewidth=2, alpha=0.8)
            
       
    h, l = [(a + b) for a, b in zip(axs[0,1].get_legend_handles_labels(), axs[0,0].get_legend_handles_labels())]

    fig.legend( h, l,
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           ncol=4
           )
    fig.subplots_adjust(bottom=0.242, hspace=0.08, top=0.92) # or whatever
    plt.show()
    #plt.savefig(name+'.png')
plotit_p()
plotit_configs()
