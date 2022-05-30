### SECLEDS

#### Installation

Tools required:
- Python3
- Cython
- [river, banditpam, minibatchkmeans] (depending on which baselines are executed)

To compile Cython code, use the following command:

> cd cython_sources/ && python setup.py build_ext --inplace


#### Running SECLEDS Clustering

1. Set-up the configuration file:

- mainconfig =  SECLEDS, SECLEDS-dtw
; Options: SECLEDS, SECLEDS-dtw
- online_baselines = SECLEDS-rand, MiniBatchKMeans
; Options: <leave empty>, SECLEDS-rand, SECLEDS-rand-dtw, MiniBatchKMeans, CluStream, StreamKM
- offline_baselines = BanditPAM
; Options: <leave empty>, BanditPAM
- plot_to_2d = False
- trials = 2
- batch_factor = 1.5
- drift = False
- drift_factor = 0.05
; 0.05#0.001 (freq)  #0.05 (phase)
- shuffle_stream = True
- skip_eval = True
- plot_extras = True
- verbose = True
- realtime_animation = False

2. To start real-time clustering with SECLEDS, use the following command:

> python secleds.py k p streamType path/to/streamFile [-N N] [-ini INI] [-h]

- k:
- p:
- streamType: {points,uni-sine,multi-chars,multi-traffic}
- path/to/streamFile:
- -N:
- -ini:
- -h: Print help

e.g.

For a uni-variate sequential dataset with 10 classes and 5 prototypes per cluster,

> python secleds.py uni-sine 10 5 datasets/sine-curve-data/

3. Print out the averages of important metrics for each clustering algorithm, run

> python average_calculator.py <path/to/experimental/folder>/exp-results.txt


4. To recreate the experimental results (and a comparison with offline k-medoids)

> python plot_paper_results.py 

#### Synthetic data creation

> python create_data.py dataType {k} {N}

- dataType: {sine-curve|blobs|circles}
- k: 
- N: 



