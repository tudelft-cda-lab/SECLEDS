# SECLEDS: Sequence Clustering in Evolving Data Streams via Multiple Medoids and Medoid Voting

SECLEDS is a real-time sequence clustering variant of the popular k-medoids algorithm that uses multiple prototypes per cluster and a prototype voting scheme. This repository accompanies our publication:

"SECLEDS: Sequence Clustering in Evolving Data Streams via Multiple Medoids and Medoid Voting" at ECML/PKDD'22.

## Installation

Libraries required:
- `python3`
- `cython`
- `dtw-python`
- {`river's CluStream and STREAMKMeans` | `banditpam` | `scikitlearn's minibatchkmeans`} (depending on which baselines are executed)
- `aiofiles` (for secleds-stream)

The SECLEDS cluster assignment and update modules are implemented using Cython. To compile that part of the code, use the following command:

> `cd cython_sources/ && python setup.py build_ext --inplace`


## Running SECLEDS Clustering

1. Set-up the configuration file: By default, it is called `config.ini`:
  - `mainconfig`:  Lists the SECLEDS configuration to be executed. 
    -  Options: {`SECLEDS`, `SECLEDS-dtw`}
  - `online_baselines`: Lists other online clustering algorithms for comparison purposes. 
    -  Options: {[leave empty], `SECLEDS-rand`, `SECLEDS-rand-dtw`, `MiniBatchKMeans`, `CluStream`, `StreamKM`}
  - `offline_baselines`: Lists offline variants of the k-medoids algorithm. 
    - Options: {[leave empty], `BanditPAM`}
  - `plot_to_2d`: Choose `True` to use t-SNE to reduce dimensionality of the dataset for visualization purposes.
  - `trials`: Number of times each algorithm is to be executed on a randomly shuffled stream.
  - `batch_factor`: Determines how big the batch is to initialize the clustering algorithms. Default value is 1.5. 
  - `drift`: Choose `True` to synthetically add drift to the input stream. Only supports sine-curves and blobs.
  - `drift_factor`: Determines how much drift to add. A default of 0.05 adds sufficient drift to the frequency of sine-curves or both dimensions of blobs. Drift can be added to the phase of the sine-curves by altering the source code, if deemed necessary.
  - `shuffle_stream`: Choose `True` to randomize the order of incoming stream data before each trial run.
  - `skip_eval`: Choose `True` to skip computing evaluation metrics, e.g., F1, cluster purity. Skipping evaluation significantly speeds up the run-time.
  - `plot_extras`: Choose `True` to plot scatter plots and other misc. graphs to help track the progress of the clustering. 
  - `verbose`: Choose `True` to print extra details regarding the clustering progress on the console.
  - `realtime_animation`: Choose `True` to have a real-time graphical view of the clustering.  

2. To start real-time clustering with SECLEDS, use the following command:

> `python secleds.py k p streamType path/to/streamFile [-N N] [-ini INI] [-h]`

- `k`: Number of clusters.
- `p`: Number of prototypes per cluster.
- `streamType`: Data type of the individual items in the stream. Choose from {points | uni-sine | multi-chars | multi-traffic}.
- `path/to/streamFile`: Path to the file(s) containing the streaming data.
- `-N`: [Optional] Limits the total number of data items to read from the stream.
- `-ini`: [Optional] Path to the configuration file. By default, it is set to `config.ini`.
- `-h`: [Optional] Prints help

e.g.

For a uni-variate sequential dataset with 10 classes and 5 prototypes per cluster, use

> `python secleds.py uni-sine 10 5 datasets/sine-curve-data/`

All the results of the clustering are stored in a folder named `[current-date-time]-plots/`

3. To print out the averages of important metrics for each clustering algorithm, run

> `python average_calculator.py [path/to/results/folder]/exp-results.txt`

4. To recreate the experimental results (and a comparison with offline k-medoids)

> `python plot_paper_results.py`

## Synthetic data creation

There is a script that allows to generate new synthetic streaming datasets with `k` separable classes. It can currently generate univariate sine-curves, and point datasets (blobs and circles).
> `python create_data.py dataType k N`

- `dataType`: Data type of the individual items in the stream. Choose from {sine-curve | blobs | circles}.
- `k`: Number of classes in the stream.
- `N`: Total items in the stream.

**If you use SECLEDS in a scientific work, consider citing the following paper:**

```
@inproceedings{nadeem2022secleds,
  title={SECLEDS: Sequence Clustering in Evolving Data Streams via Multiple Medoids and Medoid Voting},
  author={Nadeem, Azqa and Verwer, Sicco},
  booktitle={In proceedings of ECML/PKDD},
  publisher={Springer},
  year={2022}
}
```
