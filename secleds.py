import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE
import sys, os, math, copy
import time, json
from IPython import display
from collections import Counter
import itertools, configparser, argparse

# Importing source code
from cython_sources.fast_assign import exact_assign, st_exact_assign
from cython_sources.fast_update import prototypevoting_update, st_prototypevoting_update
from init_routines import perfect_init, random_init, nonuniform_init, st_nonuniform_init
from baseline_algos import Bpam_cluster
from baseline_algos import MiniBatch_init, MiniBatch_cluster
from baseline_algos import CluStream_init, CluStream_cluster
from baseline_algos import StreamKM_init, StreamKM_cluster

from helpers import format_scores, plot_votes, plot_over_time, shuffle_stream, plot_proto, plot_seq, plot_data, plot_all, plot_onlineBL, plot_offlineBL, plot_heatmap, plot_medoids, add_drift, plot_letters, prototype_distance
from cython_sources.eval_pairwise import adjacency_accuracy, evaluate_purity_complete
from evaluations import evaluate_PR,  proto_purity
from data_generation import read_curves, read_chars, read_points, read_traffic

import GLOBALS

parser = argparse.ArgumentParser(description='SECLEDS: Real-time sequence clustering via k-medoids.')
parser.add_argument('k', type=int,  help='Number of clusters')
parser.add_argument('p', type=int,  help='Number of medoids/prototypes')
parser.add_argument('datatype', type=str,  choices=['points', 'uni-sine', 'multi-chars', 'multi-traffic'], help='Datatype of the individual items in the stream')
parser.add_argument('-ini', type=str, default='config.ini', help='Path/to/configuration/file')
parser.add_argument('streamFile', type=str, help='Path/to/file/containing/stream')
parser.add_argument('-N', type=int, default=1000, help='# samples to read from stream')
args = parser.parse_args()

nclasses = args.k
nprototypes = args.p
DATASET = args.datatype
inifile = args.ini
SAVED_PATH = args.streamFile
nsamples = args.N

config = configparser.ConfigParser()
config.sections()
config.read(inifile)

secleds_version = config['ALGOS']['mainconfig'].replace(' ','').split(',')
secleds_version = list(filter(None, secleds_version))
online_baselines = config['ALGOS']['online_baselines'].replace(' ','').split(',')
online_baselines = list(filter(None, online_baselines))
offline_baselines = config['ALGOS']['offline_baselines'].replace(' ','').split(',')
offline_baselines = list(filter(None, offline_baselines))

PLOT_TO_2D = config['EXP'].getboolean('plot_to_2d')
DRIFT = config['EXP'].getboolean('drift')
drift_factor = float(config['EXP']['drift_factor'])
SKIP_EVAL = config['EXP'].getboolean('skip_eval')
PLOT_EXTRAS = config['EXP'].getboolean('plot_extras')
VERBOSE = config['EXP'].getboolean('verbose')
RT_ANIMATION = config['EXP'].getboolean('realtime_animation')
SHUFFLE_STREAM = config['EXP'].getboolean('shuffle_stream')
batch_factor = float(config['EXP']['batch_factor'])
batchsize = int(nclasses*nprototypes*batch_factor)
ntrials = int(config['EXP']['trials'])
GLOBALS.init()
now = datetime.now()
now_str = now.strftime("%d%m%y-%H%M%S")+'-plots'
os.mkdir(now_str)
fname = now_str+'/exp-results.txt'

# -------- Constant mappings
pal = sns.color_palette("hls", nclasses)
OT_pal = ['aqua', 'blue', 'green','orange', 'palevioletred', 'red', 'maroon',  'magenta', 'mediumorchid']
MET_OT = ['purity', 'complete', 'precision', 'recall', 'F1', 'mistakes',  'time-to-cluster']
BL_NAMES = ['BanditPAM', 'fasterPAM',  'MiniBatchKMeans', 'CluStream', 'StreamKM']
algorithms = {
    # SECLEDS flavors
    'SECLEDS': (nonuniform_init, st_exact_assign, st_prototypevoting_update),
    'SECLEDS-dtw': (st_nonuniform_init, exact_assign, prototypevoting_update),
    'SECLEDS-perfect-init' : (perfect_init, st_exact_assign, st_prototypevoting_update),
    'SECLEDS-perfect-init-dtw' : (perfect_init, exact_assign, prototypevoting_update),
    'SECLEDS-rand' : (random_init, st_exact_assign, st_prototypevoting_update),
    'SECLEDS-rand-dtw' : (random_init, exact_assign, prototypevoting_update),
    # online baselines
    'MiniBatchKMeans' : (MiniBatch_init, MiniBatch_cluster, MiniBatch_cluster),
    'CluStream' : (CluStream_init, CluStream_cluster, CluStream_cluster),
    'StreamKM' : (StreamKM_init, StreamKM_cluster, StreamKM_cluster),
    # offline baselines
    'BanditPAM' : (Bpam_cluster, Bpam_cluster, Bpam_cluster)
}

X, ann, labs, dist, classdict, params , metadata = None, None, None, None, {},{}, None
classes = []
#   ----------------------------------------------------------             1. Data selection
print('Reading the stream...')
if DATASET == 'points':
    classes = [x for x in range(nclasses)]
    if SAVED_PATH != '':
        (X, ann, labs, dist, classdict, metadata) = read_points(nsamples, nclasses, SAVED_PATH)
elif DATASET == 'uni-sine':
    classes = [x for x in range(nclasses)]
    if SAVED_PATH != '':
        (X, ann, labs, dist, classdict, params, metadata) = read_curves(nsamples, nclasses, SAVED_PATH)
elif DATASET == 'multi-chars':
    classes = ['C', 'U', 'V', 'W', 'S', 'O', '1', '2', '3', '5', '6', '8', '9']
    classes = classes[:nclasses]
    if SAVED_PATH != '':
        (X, ann, labs, dist, classdict, metadata) = read_chars(classes, SAVED_PATH)
elif DATASET == 'multi-traffic':
    if SAVED_PATH != '':
        (X, ann, labs, dist, classdict, metadata) = read_traffic(nclasses, SAVED_PATH)
        classes = list(classdict.values())
else:
    print('Something went wrong...')
    sys.exit(-1)

class_distro = dict({k:int(v) for k,v in Counter(labs).items()})

X_embedded = None
if PLOT_TO_2D:
    X_embedded = TSNE(random_state=42, n_components =2).fit_transform(dist)
else:
    X_embedded = X

# ----------------------------------------------------------             2. View data

print('Plotting the data stream...')
if PLOT_EXTRAS:
    plot_data(X_embedded, labs, classdict, pal, DATASET, now_str)
plot_heatmap(X, labs, classes, DATASET, now_str, 20, "", metadata)

# ----------------------------------------------------------             4. Set up experiments
print('Setting up experiments ...')
configs = []
if len(secleds_version) >= 1:
    configs.extend([(x, algorithms[x]) for x in secleds_version])
if len(online_baselines) >= 1:
    configs.extend([(x, algorithms[x]) for x in online_baselines])
if len(configs) == 0:
    print('No streaming clustering algorithms given. Exiting...')
    sys.exit()
strs = [x for x,y in configs]
trials = {key: [None]*ntrials for key in strs}
if len(offline_baselines) >= 1:
    for olbl in offline_baselines:
        trials[olbl] = [None]*ntrials
if VERBOSE:
    print('Running algos: ', strs)
outfile = open(fname, 'w')

metrics_over_time = {key: {} for key in MET_OT}
b_baseline = {key: 0.0 for key in MET_OT}
votesOT = {}
bmistakes, bpurity, bcomplete, bprecision, brecall, bf1, bPAM_end, bPAM_start, bTP, bTN, bFP, bFN, bp_purity, bc_discovered = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

### -------------------------------------------------------   Experiment loops

for trial in range(1, ntrials + 1):
    ### SHUFFLE DATA ###
    # Run experiment TRIALS times, shuffle sequences each time and  then initialize clustering and run the actual stream clustering
    print('\n----- TRIAL #', trial)
    if SHUFFLE_STREAM:
        if metadata == {}:
            group = shuffle_stream(X, ann, labs, dist, X_embedded, classes, nprototypes, batchsize, SKIP_EVAL)
            (X, ann, labs, dist, X_embedded) = zip(*group)
        else:
            group = shuffle_stream(X, zip(ann, metadata), labs, dist, X_embedded, classes, nprototypes, batchsize, SKIP_EVAL)
            (X, ann_meta, labs, dist, X_embedded) = zip(*group)
            ann, metadata = zip(*ann_meta)

    ann_new = [x for x, y in enumerate(X)]  # IDs again

    ## Apply drift
    X_exp = copy.deepcopy(X)
    X_exp_embedded = copy.deepcopy(X_embedded)
    if DRIFT:
        if DATASET == 'uni-sine' and params != {}:
            params_updated = [-1] * len(params)
            for iid, i in enumerate(ann):
                params_updated[iid] = params[i]
            # adding drift
            drifted = add_drift(DATASET, X, params_updated, drift_factor)
            X_exp = copy.deepcopy(drifted)
            X_exp_embedded = copy.deepcopy(drifted)  # or tsne()

            if PLOT_EXTRAS:
                plot_data(X_exp_embedded, labs, classdict, pal, DATASET, now_str, '-drifted-' + str(trial))
            plot_heatmap(X_exp, labs, classes, DATASET, now_str, 20, '-drifted-' + str(trial), metadata)
        elif DATASET == 'points':
            drifted = add_drift(DATASET, X, {}, drift_factor)
            X_exp = copy.deepcopy(drifted)
            X_exp_embedded = copy.deepcopy(drifted)  # or tsne()

            if PLOT_EXTRAS:
                plot_data(X_exp_embedded, labs, classdict, pal, DATASET, now_str, '-drifted-' + str(trial))


    #     1. BanditPAM:
    if 'BanditPAM' in offline_baselines:
        if VERBOSE:
            print('BanditPAM starting')
        if DATASET == 'multi-chars':
            exp_data = dist
        else:
            exp_data = X_exp

        bPAM_start = time.time()
        (points, bandit_labels, prototypes, proto_idx, meta) = Bpam_cluster(exp_data, nclasses, metadata)
        bPAM_end = time.time()
        if VERBOSE:
            print('bpam eval starting ...')

        #       Evaluate BanditPAM:
        pairs = itertools.combinations(range(len(labs)), 2)
        (bp_purity, bc_discovered) = proto_purity(proto_idx, labs, nclasses)
        if not SKIP_EVAL:
            (bpurity, bcomplete) = evaluate_purity_complete(bandit_labels, labs)
            (bTP, bTN, bFP, bFN) = adjacency_accuracy(bandit_labels, labs, pairs)
            (bprecision, brecall, bf1) = evaluate_PR(bTP, bTN, bFP, bFN)
            bmistakes = (bFP + bFN) / (bTP + bTN + bFP + bFN)

        # Plot BanditPAM
        if PLOT_EXTRAS:
            plot_offlineBL('BanditPAM', DATASET, X_exp_embedded, points, bandit_labels, pal, trial, nclasses,
                           [bf1, bpurity], now_str)
        if DATASET == 'multi-chars':
            pass  # plot_letters('BanditPAM', trial, prototypes, nclasses, 1, DATASET, now_str, meta)
        else:
            plot_medoids('BanditPAM', trial, prototypes, nclasses, 1, bp_purity, DATASET, now_str, meta)

        if VERBOSE:
            print('bpam time ', (bPAM_end - bPAM_start))
            print('bpam ended')

        b_baseline['mistakes'] += bmistakes
        b_baseline['purity'] += bpurity
        b_baseline['complete'] += bcomplete
        b_baseline['precision'] += bprecision
        b_baseline['recall'] += brecall
        b_baseline['F1'] += bf1
        b_baseline['time-to-cluster'] += (bPAM_end - bPAM_start)


    centroids = None
    total_tp, total_tn, total_fp, total_fn, init_purity, init_complete = 0, 0, 0, 0, -1, -1
    proto_dist = None
    votesOT[trial] = {}
    BL_MODEL = None

    # Run a trial with all configurations
    for config_name,(INIT, ASSIGN, UPDATE) in configs:
        fig = None
        if RT_ANIMATION:
            fig = plt.figure(figsize=(10, 10))
        GLOBALS.count_dist = 0
        _buffer = []
        func_names = (INIT.__name__, ASSIGN.__name__, UPDATE.__name__)
        print('\n!!!!!!!!!!!', 'CONFIG: ', config_name, '!!!!!!!!!!!!!')

        if PLOT_EXTRAS:
            for _i, _met in enumerate(MET_OT):
                if config_name not in metrics_over_time[_met].keys():
                    if _met == 'time-to-cluster':
                        metrics_over_time[_met][config_name] = 0.0
                    else:
                        metrics_over_time[_met][config_name] = np.array([0] * len(X_exp[batchsize:]))
        temp_metric_ = {key: np.array([None] * len(X_exp[batchsize:])) for key in MET_OT}
        temp_metric_['time-to-cluster'] = 0.0
        votesOT[trial][config_name] = {}
        for _cl in range(nclasses):
            votesOT[trial][config_name][_cl] = {key: [0] * len(X_exp[batchsize:]) for key in range(nprototypes)}

        ### +++ INIT START +++
        print('########## Init starting... ##########')
        if 'st_' in ' '.join(func_names) and DATASET in ['uni-sine', 'multi-chars']:
            (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative) = INIT(dist[0:batchsize],
                                                                                                  labs[0:batchsize],
                                                                                                  nprototypes, nclasses,
                                                                                                  classdict)
        elif config_name in BL_NAMES:
            (prototypes, proto_idx, assigned_clusters, pvotes, BL_MODEL) = INIT(X_exp[0:batchsize], labs[0:batchsize],
                                                                                nprototypes, nclasses, classdict)
        else:
            (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative) = INIT(X_exp[0:batchsize],
                                                                                                  labs[0:batchsize],
                                                                                                  nprototypes, nclasses,
                                                                                                  classdict)
        # plot the protos
        if RT_ANIMATION:
            plot_proto(fig, config_name, assigned_clusters, proto_idx, X_exp_embedded, ann_new, pal, classes)
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()
        # Start evaluation
        pairs = itertools.combinations(range(batchsize), 2)
        if not SKIP_EVAL:
            (init_purity, init_complete) = evaluate_purity_complete(assigned_clusters[:batchsize], labs[:batchsize])
            (total_tp, total_tn, total_fp, total_fn) = adjacency_accuracy(assigned_clusters[:batchsize],
                                                                      labs[:batchsize], pairs)
        ### +++ INIT END +++


        ### +++ STREAM START +++
        print('########## Stream starting... ##########')
        stream_seq = dist[batchsize:] if ('st_' in ' '.join(func_names) and DATASET in ['uni-sine', 'multi-chars']) else X_exp[batchsize:]
        _stream = zip(ann_new[batchsize:], stream_seq)
        # One sequence at a time
        loopstart = time.time()
        for zidx, (pidx, point) in enumerate(_stream):
            global_idx = pidx
            if VERBOSE:
                print('.', end=' ', flush=True)

            start_clustering, end_clustering = 0.0, 0.0
            if config_name in BL_NAMES:
                start_clustering = time.time()  # Time start
                (assigned_clusters, pvotes, BL_MODEL) = ASSIGN(prototypes, point, assigned_clusters, pvotes,
                                                               BL_MODEL)
                end_clustering = time.time()  # Time end
                if 'MiniBatch' in config_name:
                    centroids = BL_MODEL.cluster_centers_
                else:
                    centroids = [(y[0], y[1]) for x, y in BL_MODEL.centers.items()]
            else:
                start_clustering = time.time()  # Time start

                # Cluster assignment phase: assigning cluster to a point
                (minimum_idx, assigned_clusters, pvotes) = ASSIGN(prototypes, point, assigned_clusters, proto_dist,
                                                                  pvotes, representative)
                # Cluster update phase: replacing a prototype
                (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, repl) = UPDATE(prototypes,
                                                                                                    point,
                                                                                                    minimum_idx,
                                                                                                    proto_idx,
                                                                                                    global_idx,
                                                                                                    proto_dist,
                                                                                                    pvotes,
                                                                                                    representative,
                                                                                                    _buffer)
                end_clustering = time.time()  # Time end

                if not SKIP_EVAL:
                    for __c in range(nclasses):
                        for __p in range(nprototypes):
                            votesOT[trial][config_name][__c][__p][zidx] = pvotes[__c][__p]
            if len(assigned_clusters[batchsize:]) > 0:
                _ttc = (end_clustering - start_clustering)
                temp_metric_['time-to-cluster'] += _ttc


                pur, com, pre, rec, _f1, tot = -1, -1, -1, -1, -1, -1
                

                if not SKIP_EVAL:
                    (pur, com) = evaluate_purity_complete(assigned_clusters[batchsize:], labs[batchsize:pidx + 1])
                    (tp, tn, fp, fn) = adjacency_accuracy(assigned_clusters, labs, pidx, batchsize)

                    total_tp += tp
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn
                    (pre, rec, _f1) = evaluate_PR(total_tp, total_tn, total_fp, total_fn)

                c = total_tp + total_tn
                m = total_fp + total_fn
                tot = c + m

                temp_metric_['mistakes'][zidx] = m / tot if tot > 0 else -1
                temp_metric_['purity'][zidx] = pur
                temp_metric_['complete'][zidx] = com
                temp_metric_['precision'][zidx] = pre
                temp_metric_['recall'][zidx] = rec
                temp_metric_['F1'][zidx] = _f1

            # plot everything
            if RT_ANIMATION:
                plot_seq(fig, config_name, assigned_clusters, proto_idx, X_exp_embedded, ann_new, pal,
                         classes, pvotes)
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
        loopend = time.time()

        if VERBOSE:
            print('\nClustering loop took', (loopend-loopstart))
        if PLOT_EXTRAS:
            for _m_, mname in enumerate(MET_OT):
                if mname == 'time-to-cluster':
                    metrics_over_time[mname][config_name] += temp_metric_[mname]
                else:
                    metrics_over_time[mname][config_name] = metrics_over_time[mname][config_name] + temp_metric_[mname]
        ### +++ STREAM ENDS +++

        ### +++ EVALUATE +++
        time_to_cluster = temp_metric_['time-to-cluster']
        if VERBOSE:
            print('Internal clustering time ', time_to_cluster)

        purity, complete, p_purity, c_discovered, precision, recall, f1, str_true, str_pred = -1, -1, -1, -1, -1, -1, -1, '', ''
        if config_name not in BL_NAMES:
            (p_purity, c_discovered) = proto_purity(proto_idx, labs, nclasses)
        
        if not SKIP_EVAL:
            (purity, complete) = evaluate_purity_complete(assigned_clusters[batchsize:], labs[batchsize:])
            (precision, recall, f1) = evaluate_PR(total_tp, total_tn, total_fp, total_fn)
        str_true = copy.deepcopy(labs)
        str_true = '|'.join([str(x) for x in str_true])
        str_pred = copy.deepcopy(assigned_clusters)
        str_pred = [-1 if x is None else x for x in str_pred]
        str_pred = '|'.join([str(x) for x in str_pred])
        tupple = (round(init_purity, 4), round(init_complete, 4), round(purity, 4), round(complete, 4),
                  round(precision, 4), round(recall, 4), round(f1, 4), round(p_purity, 4),
                  round(c_discovered, 4), total_tp, total_tn, total_fp, total_fn, GLOBALS.count_dist,
                  round(time_to_cluster, 4), dict(Counter(assigned_clusters).items()), class_distro, DRIFT, drift_factor, str_pred,
                  str_true)
        trials[config_name][trial - 1] = tupple

        if VERBOSE:
            print('eval calc ended')


        ### +++ PLOT CLUSTERING RESULT +++
        if config_name in BL_NAMES:
            if PLOT_EXTRAS:
                plot_onlineBL(config_name, DATASET, X_exp_embedded, centroids, assigned_clusters, pal, trial, nclasses,
                          [f1, purity], now_str)
        else:
            if PLOT_EXTRAS:
                plot_all(config_name, (trial, nclasses, nprototypes), DATASET,
                         (f1, init_purity, purity, p_purity, c_discovered),
                         assigned_clusters, proto_idx, X_exp_embedded, ann_new, pal, classes, pvotes, now_str)
                ### Plotting final medoids
            if metadata != {}:
                meta = [[metadata[p] for p in prot] for prot in proto_idx]
            else:
                meta = None
            if DATASET == 'multi-chars':
                pass #plot_letters(config_name, trial, prototypes, nclasses, nprototypes, DATASET, now_str, meta)
            else:
                plot_medoids(config_name, trial, prototypes, nclasses, nprototypes, p_purity, DATASET, now_str, meta)
        print('plotting done')
    if 'BanditPAM' in offline_baselines:
        str_true, str_pred = '', ''
        #if not SKIP_EVAL:
        str_true = copy.deepcopy(labs)
        str_true = '|'.join([str(x) for x in str_true])
        str_pred = copy.deepcopy(bandit_labels)
        str_pred = [-1 if x is None else x for x in str_pred]
        str_pred = '|'.join([str(x) for x in str_pred])

        trials[('BanditPAM')][trial - 1] = (
        -1, -1, round(bpurity, 4), round(bcomplete, 4), round(bprecision, 4), round(brecall, 4), round(bf1, 4),
        round(bp_purity, 4), round(bc_discovered, 4), bTP, bTN, bFP, bFN, -1, round((bPAM_end - bPAM_start), 4),
        dict(Counter(bandit_labels).items()), class_distro, DRIFT, drift_factor, str_pred, str_true)

print('Plotting final stuff')

if PLOT_EXTRAS:
    if not SKIP_EVAL:
        plot_votes(votesOT, nclasses, now_str)
    offline_baseline = {}
    for _m_, mname in enumerate(MET_OT):
        if 'BanditPAM' in offline_baselines:
            offline_baseline = {"BanditPAM": b_baseline[mname], }
            # "fasterPAM":f_baseline[mname]}

        plot_over_time(mname, metrics_over_time[mname], offline_baseline, BL_NAMES, OT_pal, ntrials, now_str)

print('Writing output')
### +++ EVAL START +++
perfs = format_scores(trials, SAVED_PATH, len(X), ntrials, batchsize, len(X[0]), nprototypes, nclasses, classdict, VERBOSE)
jj = json.dumps(perfs, indent=4)
outfile.write(jj)
outfile.close()

### +++ EVAL END +++
