import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
#from pyclustering.cluster.kmedoids import kmedoids

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
import sys, os, re, glob, math
import scipy
from sklearn import metrics
from collections import Counter
import itertools
import GLOBALS 

def purity_score(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    
    dominance = np.amax(contingency_matrix, axis=0)
    cluster_sizes = np.sum(contingency_matrix, axis=0)
    purity = np.sum(dominance/cluster_sizes)/contingency_matrix.shape[1]
    
    # shape = (m,n) where m is rows [labels], n is cols [clusters] 

    return purity

def completeness_score(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    
    label_dominance = np.amax(contingency_matrix, axis=1)
    label_sizes = np.sum(contingency_matrix, axis=1)
    complete = np.sum(label_dominance/label_sizes)/contingency_matrix.shape[0]
    
    return complete

def proto_spread(proto_dist):
    avg_clus = []
    for prototypes in proto_dist: # for each cluster 'cluster'
        avg_clus.append(round(sum(prototypes.values())/float(len(prototypes)), 2))  
    return avg_clus  
    

def proto_purity(proto_idx, labels, nclasses): #prototidx, labs, nclasses
    final_purity = 0.0
    classes_discovered = 0.0
    classes = set()
    for cluster in proto_idx:
        gt = [labels[prototype_id] for prototype_id in cluster]
        classes.update(gt)
        counts = dict(Counter(gt))
        assert len(gt) == sum(counts.values())
        cluster_purity = max(counts.values())/float(len(gt)) # or sum(counts.values())
        final_purity += cluster_purity
        
    final_purity = final_purity/float(nclasses)
    classes_discovered = len(classes)/float(nclasses)
    
    
    return (final_purity, classes_discovered)
    

               
def adjacency_accuracy(labels, predictions, pairs, bs=-1):

    TP, TN, FP, FN = 0, 0, 0, 0
    sameclass, samecluster = 0, 0

    if isinstance(pairs, int):
        pairs = [(x,pairs) for x in range(pairs)]
        
    for pair in pairs:
        if bs > -1 and (pair[0] < bs or pair[1] < bs):
            continue # in case of ids from batch, we dont evaluate
        else:
            gt = (labels[pair[0]] == labels[pair[1]])
            pred = (predictions[pair[0]]== predictions[pair[1]])
            if gt and pred:
                TP += 1
            elif gt and not pred:
                FN += 1
            elif not gt and pred:
                FP += 1
            elif not gt and not pred: 
                TN += 1

    return (TP, TN, FP, FN)    

def evaluate_PR(TP, TN, FP, FN):
    precision = TP/(float(TP)+float(FP)) if (TP+FP) > 0 else 0.0
    recall = TP/(float(TP)+float(FN)) if (TP+FN) > 0 else 0.0
    f1 = 0.0
    try:
        f1 = 2*((precision*recall)/(precision+recall))
    except:
        f1 = 0.0
    
    return (precision, recall, f1)

# Main evaluation method    
def evaluate_purity_complete(assigned_clusters, labs):    
    
    Nones = [i for i, x in enumerate(assigned_clusters) if x is None]
    y_pred = np.array([x for i,x in enumerate(assigned_clusters) if i not in Nones])
    y_true = np.array([x for i,x in enumerate(labs) if i not in Nones])
    
    
    purity = purity_score(y_true, y_pred)
    complete = completeness_score(y_true, y_pred)
    return (purity, complete)

def evaluate(assigned_clusters, labs, classes, _init):
    Nones = [i for i, x in enumerate(assigned_clusters) if x is None]
    
    y_pred = np.array([x for i,x in enumerate(assigned_clusters) if i not in Nones])
    y_true = np.array([x for i,x in enumerate(labs) if i not in Nones])
    
    
    purity = purity_score(y_true, y_pred)
    complete = completeness_score(y_true, y_pred)
    pairs = itertools.combinations(range(len(y_pred)), 2)
    (TP, TN, FP, FN) = adjacency_accuracy(y_true, y_pred, pairs)
    (precision, recall, f1) = evaluate_PR(TP, TN, FP, FN)

    return (TP, TN, FP, FN, purity, complete, precision, recall, f1)
    
