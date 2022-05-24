#!python
#cython: language_level=3

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
from sklearn import metrics
from dtw import dtw
import math
import GLOBALS 

def adjacency_accuracy(labels, predictions, pairs, bs=-1):

    cdef long long int TP, TN, FP, FN
    TP = 0
    TN = 0
    FP = 0
    FN = 0

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
    

def purity_score(y_true, y_pred):
    cdef float purity

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
    cdef float complete

    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    
    label_dominance = np.amax(contingency_matrix, axis=1)
    label_sizes = np.sum(contingency_matrix, axis=1)
    complete = np.sum(label_dominance/label_sizes)/contingency_matrix.shape[0]
    
    return complete
    
# Main evaluation method    
def evaluate_purity_complete(assigned_clusters, labs, init=True):    
    
    y_pred, y_true = None, None
    if init:
        Nones = [i for i, x in enumerate(assigned_clusters) if x is None]
        y_pred = np.array([x for i,x in enumerate(assigned_clusters) if i not in Nones])
        y_true = np.array([x for i,x in enumerate(labs) if i not in Nones])
    else:
        y_pred = assigned_clusters
        y_true = labs
        
    purity = purity_score(y_true, y_pred)
    complete = completeness_score(y_true, y_pred)
    return (purity, complete)
    

    
    
    
    

