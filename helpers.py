import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#from pyclustering.cluster.kmedoids import kmedoids
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
from dtw import dtw
import sys, os, re, glob, math
import scipy
import GLOBALS 
from IPython import display
import time
from matplotlib.patches import Rectangle
from data_generation import generateCurve


def shuffle_stream(X, ann, labs, dist, X_embedded, classes, nprototypes, batchsize, SKIP_EVAL):
    ## Shuffle stream
    group = [(u, w, x, y, z) for (u, w, x, y, z) in zip(X, ann, labs, dist, X_embedded)]
    ## We cannot assume to know how big each class is in stream setting.
    # We can assume that points from each class are provided s.t. enough protos can be selected
    # TODO: rethink when considering drift
    print('Shuffling...')
    init_batch = [None] * batchsize  # [None]*(nprototypes*len(classes))
    selected_ = [None] * batchsize  # [None]*(nprototypes*len(classes))

    if SKIP_EVAL:
        init_batch = group[0:batchsize]
        stream = group[batchsize:]
    else:
        for cid, class_ in enumerate(classes):
            gp = [(i, y) for (i, (x, y)) in enumerate(zip(labs, group)) if x == class_]
            if len(gp) == 0:
                continue
            gp = random.sample(gp, nprototypes)
            sel, gp = zip(*gp)
            start = cid * nprototypes
            end = start + nprototypes
            init_batch[start:end] = gp
            selected_[start:end] = sel  # already selected for protos

        # select extras
        still_tobe_selected = abs(batchsize - (nprototypes * len(classes)))
        if still_tobe_selected > 0:
            gp = [(i, y) for (i, (x, y)) in enumerate(zip(labs, group)) if i not in selected_]
            gp = random.sample(gp, still_tobe_selected)
            sel, gp = zip(*gp)
            init_batch[(nprototypes * len(classes)): batchsize] = gp
            selected_[(nprototypes * len(classes)): batchsize] = sel
            # print('selected', selected_)
        # break into batch and stream

        stream = [g for i, g in enumerate(group) if i not in selected_]

    # print('stream has %d items'%(len(stream)))
    # shuffle the stream (and batch?)
    random.shuffle(stream)
    random.shuffle(init_batch)

    # put them back together
    group = [None] * len(X)
    group[:batchsize] = init_batch
    group[batchsize:] = stream

    return group

def add_drift(dataset, data, params, drift):
    drifted_ = []
    if dataset == 'uni-sine':
        (newfreq, newerr, newphase) = zip(*params)
        for cid in range(len(data)):
            newcurve =  generateCurve(1, [newfreq[cid]], newerr[cid], newphase[cid]+(drift*cid))[0][0]
            drifted_.append(newcurve)   
    elif dataset == 'points':
        for did, point in enumerate(data):
            newpt = (point[0]+(drift*did), point[1]+(drift*did)) # adding drift in both direction
            #x = point[0]+(drift*did)
            #y = point[1]
            #a = 30
            #xnew = (x*math.cos(a)) - (y*math.sin(a))
            #ynew = (x*math.sin(a)) + (y*math.cos(a))
            drifted_.append(newpt)
            #print('orig', point, 'new', newpt)
    else:
        print('Dataset not supported for drift')
        sys.exit()
    print('Drifted %d samples'%(len(drifted_)))
    return drifted_


def plot_votes(votesOT, nclasses, now_str):
    pal_v = ['green', 'orange', 'purple', 'blue', 'red']
    for trial, confs in votesOT.items():
        for conf, clstrs in confs.items():
            if 'SECLEDS' not in conf:
                continue
            params = {'font.size': 16}

            plt.rcParams.update(params)
            # fig, ax = plt.subplots(nrows=nclasses, ncols=1, sharex=False, figsize=(15,5))

            for clstrid, clstr in clstrs.items():
                fig = plt.figure(figsize=(15, 5))
                ax = plt.gca()
                # ax[clstrid].set_title('Votes over time (Cluster=%d)'%(clstrid+1))
                # print('Cluster ', clstrid)
                for medid, med in clstr.items():
                    ax.plot(range(len(med)), med, linestyle='-', linewidth=2, label='Medoid-' + str(medid + 1),
                            color=pal_v[medid])

                ax.set(xlabel="Time", ylabel="Votes")
                ax.grid(True)
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                          mode="expand", borderaxespad=0, ncol=5)
                fig.subplots_adjust(bottom=0.25, left=0.1, right=0.98)  # or whatever
                plt.savefig(now_str + '/VotesOT-' + conf + '-trial' + str(trial) + '-k' + str(clstrid) + '.png')
def plot_over_time(met_name, struct, offline_baseline, bl_names, OT_pal, TRIALS, now_str):
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(10, 10))
    save = True
    if met_name in ['time-to-cluster']:
        if dict(offline_baseline):
            num_items = len(struct) + len(offline_baseline)
        else:
            num_items = len(struct) 
        labs = [None] * num_items
        times = [None] * num_items
        i = 0
        plt.title(met_name)
        plt.xlabel('Configs')
        plt.ylabel(met_name + ' (s)')
        labs[:len(struct)] = [config for config in struct.keys()]
        times[:len(struct)] = [x / TRIALS for x in struct.values()]

        for cidx, (c, val) in enumerate(offline_baseline.items()):
            labs[cidx + len(struct)] = c
            times[cidx + len(struct)] = val / TRIALS

        plt.bar(labs, times, color=OT_pal)
        plt.xticks(rotation=45)
        plt.tight_layout()

    else:
        plt.title(met_name + ' over time')
        plt.xlabel('Time')
        plt.ylabel(met_name)
        observations = 0
        for idx, (c, st) in enumerate(struct.items()):
            st = np.array([m / TRIALS for m in st])

            if np.sum(st) == -1 * len(st):
                save = False
                continue
            if c in bl_names:
                plt.plot(range(len(st)), st, label=c, color=OT_pal[idx], linestyle='dashed')
            else:
                plt.plot(range(len(st)), st, label=c, color=OT_pal[idx])
            observations = len(st)
        for idx, (c, val) in enumerate(offline_baseline.items()):
            st = val / TRIALS
            plt.plot(range(observations), [st] * observations, label=c, color=OT_pal[len(struct) + idx],
                     linestyle='dashed')
        plt.legend()
        plt.grid(True)
    if save:
        plt.savefig(now_str + '/' + met_name + '_over_time.png')
    plt.close(fig)

def format_scores(trials, dataset, nsamples, ntrials, batchsize, dim, nprototypes, nclasses, classdict, VERBOSE):
    print('~~~~~~~~~ Formatting all evaluations after %d runs ~~~~~~~~~'%(ntrials))
    perfs = []
    for key,measures in trials.items():
        if VERBOSE:
            print('Algo: ', key)
        config = key
        (init_purity, init_complete, purity, complete, precision, recall, f1, p_pur, c_disc, TP, TN, FP, FN,
         dists, time_to_cluster, dassigned_clusters, dlabs, drift, drift_factor, pred_labels, real_labels) = zip(*measures)
        #print([(a+b+c+d) for (a,b,c,d) in zip(TP, TN, FP, FN)])
        perf_metrics = {
                    'init' : init_purity,
                    'init_c': init_complete,
                    'purity' : purity,
                    'complete': complete,
                    'precision' : precision,
                    'recall' : recall,
                    'f1' : f1,
                    'proto_purity': p_pur,
                    'clusters_discovered':c_disc,
                    'TP': TP,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN,
                    'dists' : dists,
                    'runtime' :  time_to_cluster,
                    'runs' : ntrials,
                    'predicted_labs': pred_labels,
                    'true_labs': real_labels
                    }

        perf = {}

        perf['dataset'] = str(dataset)
        perf['config'] = config,
        perf['cluster_properties'] = {
            'batchsize': batchsize,
            'clustersize': dassigned_clusters
        }

        perf['data_properties'] = {
            'nsequences': nsamples,
            'dim': dim,
            'nclasses': nclasses,
            'nprototypes': nprototypes,
            'drift': drift,
            'drift_factor': drift_factor,
            'classes': dict(classdict.items()),
            'class_distro': dlabs
        }
        perf['metrics'] = perf_metrics
        perfs.append(perf)

    return perfs
def prototype_distance(prototypes, dist_matrix=None, partial=False, newID=-1):
    if len(prototypes) == 1:
        return {(0,0): 0.0}     
    if partial:
        for idx1, prototype1 in enumerate(prototypes):
            for idx2, prototype2 in enumerate(prototypes):
                if idx1 >= idx2:
                    continue   
                if newID in [idx1,idx2]:
                   dist = euclidean_distances(np.array(prototype1).reshape(1, -1),np.array(prototype2).reshape(1, -1))[0][0]    
                   dist_matrix[(idx1,idx2)] = dist
                   dist_matrix[(idx2,idx1)] = dist
             
    else:
        dist_matrix = {}
        for idx1, prototype1 in enumerate(prototypes):
            for idx2, prototype2 in enumerate(prototypes):
                if idx1 >= idx2:
                    continue
                dist = euclidean_distances(np.array(prototype1).reshape(1, -1),np.array(prototype2).reshape(1, -1))[0][0]
                dist_matrix[(idx1,idx2)] = dist
                dist_matrix[(idx2,idx1)] = dist
    return dist_matrix
    
def calc_represent(prototypes):
    global count_dist
    # computing representatives
    representative = []
    for protos in prototypes:
        rep_clus = []
        for p1 in range(len(protos)):
            d = 0
            for p2 in range(len(protos)):
                if p1 == p2:
                    continue
                d += dtw(protos[p1], protos[p2], distance_only=True, dist_method="euclidean").distance
                count_dist += 1
            d = d/float(len(protos)-1)
            rep_clus.append(d)
        ravg = sum(rep_clus)/float(len(rep_clus))
        
        rep_clus = [ravg/(2*x) for x in rep_clus] # SD_avg/2*SD_this # Is there a better way?
        
        representative.append(rep_clus)
    return representative

# Plot only prototypes    
def plot_proto(fig, config_name, assigned_clusters, proto_idx, X_embedded_, ann, pal, classes):
    #fig = plt.figure(figsize=(10,10))
    plt.title('RT clustering [init= '+config_name+']')
    
    protoidx = [item for sublist in proto_idx for item in sublist]
    X_embedded = np.array([X_embedded_[x] for x in protoidx])
    
    # plotting prototypes
    for i, txt in enumerate(protoidx):
        if GLOBALS.DEBUG:
            print(i, txt, 'annotate', classes[assigned_clusters[txt]])
        if assigned_clusters[txt] is None:
            continue
        plt.plot(X_embedded[i][0], X_embedded[i][1], marker= "o", color=pal[assigned_clusters[txt]])
        #plt.annotate(txt, (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[i]]) #plot seqID
        plt.annotate(classes[assigned_clusters[txt]], (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]]) #plot class label

    # plotting all others
    for i, txt in enumerate(ann):    
        plt.plot(X_embedded_[i][0], X_embedded_[i][1], marker= "o", color='black', alpha=0.2)
        #plt.annotate(txt, (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]], alpha=0.2)
        #plt.annotate(classes[assigned_clusters[txt]], (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]], alpha=0.5)

    #plt.show()

# Plot all sequences    
def plot_seq(fig, config_name, assigned_clusters, proto_idx, X_embedded_, ann, pal, classes, pvotes):
    
    #fig = plt.figure(figsize=(10,10))
    plt.title('RT clustering [stream= '+config_name+']')
    
    display.clear_output(wait=True)
    #time.sleep(1) # change the rate of rendering
    
    # Plotting prototypes
    protoidx = [item for sublist in proto_idx for item in sublist]
    X_embedded =  np.array([X_embedded_[x] for x in protoidx])
    
    
    for i, txt in enumerate(protoidx):
        if assigned_clusters[txt] is None:
            continue
        clid = i%len(pvotes[0])
        plt.plot(X_embedded[i][0], X_embedded[i][1], marker= "o", color=pal[assigned_clusters[txt]])
        #plt.annotate(txt, (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters_[txt]])
        #plt.annotate(classes[assigned_clusters[txt]], (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]]) # annotate proto with label of class
        plt.annotate(str(classes[assigned_clusters[txt]])+'/'+str(clid)+'/'+str(pvotes[assigned_clusters[txt]][clid]), (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]])
        #print()
    
    # plotting the rest
    X_embedded = X_embedded_[0:len(assigned_clusters)]
    ann_ = ann[0:len(assigned_clusters)]

    for i, txt in enumerate(ann_):
        if assigned_clusters[txt] is not None:
            plt.plot(X_embedded[i][0], X_embedded[i][1], marker= "o", color=pal[assigned_clusters[txt]], alpha=0.2)
            #plt.annotate(txt, (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters_[i]], alpha=0.2)
            #plt.annotate(classes[assigned_clusters_[txt]], (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters_[i]], alpha=0.5)

    #plt.show()

# Plot all together    
def plot_all(config_name, params, dataset, metr, assigned_clusters, proto_idx, X_embedded_, ann, pal, classes, pvotes, now_str):
    
    fig = plt.figure(figsize=(10,10))
    plt.title(config_name+' [Trial='+str(params[0])+', classes='+str(params[1])+', prototypes='+str(params[2])+']\n'+\
    '[F1=%.2f, Purity_init=%.2f, Purity_all=%.2f, \nPurity_proto=%.2f, C_found=%.2f]'%(metr))

    # Plotting prototypes
    protoidx = [item for sublist in proto_idx for item in sublist]
    X_embedded =  np.array([X_embedded_[x] for x in protoidx])
    
    
    for i, txt in enumerate(protoidx):
        clid = i%len(pvotes[0])
        #print(classes, i, clid, len(assigned_clusters), len(pal), len(pvotes))
        plt.plot(X_embedded[i][0], X_embedded[i][1], marker= "o", color=pal[assigned_clusters[txt]], markersize=10)
        #plt.annotate(classes[assigned_clusters[txt]], (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]], fontsize=12) # annotate proto with label of class
        #plt.annotate(str(classes[assigned_clusters[txt]])+'/'+str(clid)+'/'+str(round(pvotes[assigned_clusters[txt]][clid],2)), (X_embedded[i][0], X_embedded[i][1]), color=pal[assigned_clusters[txt]],fontsize=12)
        
    # plotting the rest
    for i, txt in enumerate(ann):
        if txt in protoidx:
            continue
        if assigned_clusters[txt] is not None:
            plt.plot(X_embedded_[i][0], X_embedded_[i][1], marker= "o", color=pal[assigned_clusters[txt]], alpha=0.2)
            #plt.annotate(classes[assigned_clusters[txt]], (X_embedded_[i][0], X_embedded_[i][1]), color=pal[assigned_clusters[txt]], alpha=0.5)
            
    plt.savefig(now_str+'/'+config_name+'-'+dataset+'-'+str(params[0])+'.png')
    plt.close(fig)
    return

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

# plot all data with GT
def plot_data(X_embedded, labs, classdict, pal, dataset, now_str, name = ""):
    params = {'font.size': 16 } 
             
    plt.rcParams.update(params)
            
    fig = plt.figure(figsize=(10,10))
    plt.title('Input data w/ labels ['+dataset+name+']')
    ax = plt.gca()
    for i in range(len(X_embedded)):
        tupple = X_embedded[i]
        if type(tupple[0]) == tuple:
            tupple = [x[0] for x in tupple] # Todo: handle extra dimensions
        if labs[i] is None:
            continue
        plt.plot(tupple[0],tupple[1], marker= "o", color=pal[classdict[labs[i]]], alpha=0.5, label=labs[i])
        #plt.annotate(labs[i], (tupple[0], tupple[1]), color=pal[classdict[labs[i]]], alpha=0.2)
    legend_without_duplicate_labels(ax)
    # Color = class, annotation = Sequence ID
    plt.savefig(now_str+'/'+'input-'+dataset+name+'-labeled.png')  
    plt.close()

    fig = plt.figure(figsize=(10,10))
    plt.title('Input data ['+dataset+name+']')
    
    for i in range(len(X_embedded)):
        tupple = X_embedded[i]
        if type(tupple[0]) == tuple:
            tupple = [x[0] for x in tupple] # Todo: handle extra dimensions
        plt.plot(tupple[0], tupple[1], marker= "o", color='blue')
    
    plt.savefig(now_str+'/'+'input-'+dataset+name+'.png')
    plt.close(fig)    
    return

# plot all data with GT
def plot_heatmap(X, labs, classes, dataset, now_str, max_each_class=20, name="", meta=None):
    sampleIDs = {}
    samplesize = 5 #int(random.uniform(5, max_each_class))
    seqlength = 0
    dim = 1
    checked = False
    params = {'font.size': 16 } 
             
    plt.rcParams.update(params)
    for i, cclass in enumerate(classes):
        
        thislab = [x for x,y in enumerate(labs) if y == cclass]
        if len(thislab) == 0:
            continue
            
        if len(thislab) < max_each_class:
            samplesize = len(thislab)

        sampleIDs[i] = random.sample(thislab, samplesize) #thislab[:samplesize]#
        seqlength = max(max([len(X[x]) for x in sampleIDs[i]]), seqlength)
        if not checked and type(X[sampleIDs[i][0]][0]) == tuple:
            dim = len(X[sampleIDs[i][0]][0])
            checked = True
    
    sampled = [None]*(sum([len(x) for x in sampleIDs.values()]))
    identifiers= [None]*(sum([len(x) for x in sampleIDs.values()]))
    
    
    for fid in range(dim):
        counter = 0
        for cid, samples in sampleIDs.items():
            for sample in samples:
                if type(X[sample][0]) == tuple:
                    seq = [x[fid] for x in X[sample]] 
                else:
                    seq = X[sample]
                seq = (seq + seqlength * [-1])[:seqlength]
                sampled[counter] = seq
                if meta != None:
                    identifiers[counter] = meta[sample]
                else:
                    identifiers[counter] = sample
                counter += 1
        fig = plt.figure(figsize=(10,8))
        plt.title('Temporal heatmap [%s] \nFeature [%d]\nSampling %d items per class'%(dataset, fid, samplesize))
        df = pd.DataFrame(sampled, index=identifiers)
        ax = sns.heatmap(df, center=0.0)
        plt.setp(ax.get_yticklabels(),rotation=0)
        

        borders = [samplesize*i for i in range(1, len(classes))]
        for border in borders:
            ax.add_patch(Rectangle((0, border), seqlength, 0, ec='white', fc='none', lw=1.5))
        plt.xlabel('Time/Dimensions')
        plt.ylabel('Samples')
        #plt.show()
        plt.savefig(now_str+'/'+'input-'+dataset+name+'-feature'+str(fid)+'-temporal.png')
        plt.close(fig)    
    return    
    
# plot medoids
def plot_medoids(config_name, trial, prototypes, nclasses, nprototypes, p_purity, dataset, now_str, meta=None):
    params = {'font.size': 16 } 
             
    plt.rcParams.update(params)
    samplesize = nprototypes

    sampled_ = [item for sublist in prototypes for item in sublist]
    if meta != None:
        identifiers = [item for sublist in meta for item in sublist]
    else:
        identifiers = [x for x in range(len(sampled_))]
        
    seqlength = max([len(x) for x in sampled_])
    sampled__ = []
    handle_tuple = False
    # check dimensions
    
    if type(sampled_[0][0]) == tuple:
        for i in range(len(sampled_[0][0])):
            sampled__.append([[x[i] for x in seq] for seq in sampled_]) # Todo: handle extra dimensions
        handle_tuple = True
    else:
        sampled__ = [sampled_]
    
    for fid, sampled in enumerate(sampled__):
        # check padding needs    
        if min([len(x) for x in sampled]) != seqlength:
            sampled = [[(seq + seqlength * [-1])[:seqlength] for seq in seq_] for seq_ in sampled] 
     
        
        fig = plt.figure(figsize=(10,5))
        
        #plt.title('Final medoids [%s][Purity_proto=%.2f]\n[trial=%d, classes=%d, prototypes=%d]'%(config_name, p_purity, trial, nclasses, nprototypes))
        plt.title('Final medoids [feature=%d, run=%d, classes=%d, prototypes=%d]'%(fid, trial, nclasses, nprototypes))	
        
        df = pd.DataFrame(sampled, index=identifiers)
        ax = sns.heatmap(df, center=0.0)
        plt.setp(ax.get_yticklabels(),rotation=0)
        
        borders = [samplesize*i for i in range(1, nclasses)]
        for border in borders:
            ax.add_patch(Rectangle((0, border), seqlength, 0, ec='white', fc='none', lw=0.5))
        plt.xlabel('Time/Dimensions')
        plt.ylabel('Sample ID')
        #plt.show()
        plt.savefig(now_str+'/'+'medoids-'+config_name+'-'+dataset+'-trial'+str(trial)+'-feature'+str(fid)+'.png')
        plt.close(fig)    
    return 
    
    
# plot medoids
def plot_letters(config_name, trial, prototypes, nclasses, nprototypes, dataset, now_str, meta=None):

    samplesize = nprototypes

    sampled_ = [item for sublist in prototypes for item in sublist]
    if meta != None:
        identifiers = [item for sublist in meta for item in sublist]
    else:
        identifiers = [x for x in range(len(sampled_))]


    fig = plt.figure(figsize=(10,5))
    
    plt.title('Final medoids [%s]\n[trial=%d, classes=%d, prototypes=%d]'%(config_name, trial, nclasses, nprototypes)) 
    
    for med in sampled_:
    	plt.plot([x[0] for x in med], [x[1] for x in med])
    plt.savefig(now_str+'/'+'letters-'+config_name+'-'+dataset+'-'+str(trial)+'.png')
    plt.close(fig)    
    return        
        
# Plotting baselines final clustering
def plot_onlineBL(configname, dataset, X_embedded, centers, assigned_clusters, pal, trial, nclasses, metrics, now_str):
    fig = plt.figure(figsize=(10,10))
    for p_idx in range(len(X_embedded)):
        plt.plot(X_embedded[p_idx][0], X_embedded[p_idx][1], marker= "o", color=pal[assigned_clusters[p_idx]], alpha=0.2)
    for p_idx, centroid in enumerate(centers):
        plt.plot(centroid[0], centroid[1], marker= "o", color=pal[p_idx], markersize=10)
    plt.title(configname+' [Trial=%d, classes=%d][F1=%.2f, Purity=%.2f]'%(trial, nclasses, metrics[0],metrics[1]))
    plt.savefig(now_str+'/'+configname+'-'+dataset+'-'+str(trial)+'.png') 
    plt.close(fig)
    
def plot_offlineBL(configname, dataset, X_embedded, points, assigned_clusters, pal, trial, nclasses, metrics, now_str):
    fig = plt.figure(figsize=(10,10))
    for p_idx in range(len(X_embedded)):
        if p_idx in map(int, points):
            plt.plot(X_embedded[p_idx][0], X_embedded[p_idx][1], marker= "o", color=pal[assigned_clusters[p_idx]], markersize=10)
        else:
            plt.plot(X_embedded[p_idx][0], X_embedded[p_idx][1], marker= "o", color=pal[assigned_clusters[p_idx]], alpha=0.2)
    plt.title(configname+' [Trial=%d, classes=%d][F1=%.2f, Purity=%.2f]'%(trial, nclasses, metrics[0],metrics[1]))
    plt.savefig(now_str+'/'+configname+'-'+dataset+'-'+str(trial)+'.png')
    plt.close(fig)
    
    
    
    
'''def plot_fasterPAM(X_embedded, prototypes, assigned_clusters, pal, trial, metrics, now_str):
    fig = plt.figure(figsize=(10,10))
    for p_idx in range(len(X_embedded)):
        if p_idx in map(int, prototypes):
            plt.plot(X_embedded[p_idx][0], X_embedded[p_idx][1], color=pal[assigned_clusters[p_idx]], markersize=10)
        else:
            plt.plot(X_embedded[p_idx][0], X_embedded[p_idx][1], color=pal[assigned_clusters[p_idx]], alpha=0.2)
    plt.title('fasterPAM [Trial=%d][F1=%.2f, Purity=%.2f, Comp=%.2f]'%(trial, metrics[0],metrics[1],metrics[2]))
    plt.savefig(now_str+'/BL-fasterpam-clustered-data-'+str(trial)+'.png')'''
    

