#!python
#cython: language_level=3

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from dtw import dtw
import GLOBALS 
import statistics
from scipy.spatial.distance import cdist

# IMPL4: Baseline with exact aggregated distance calculation
def aggr_exact_assign(prototypes, point, assigned_clusters, proto_dist, pvotes, _):
    cdef float minimum_distance = float("inf")
    cdef long long int minimum_idx = -1
    cdef long int cp_idx = -1 # closest proto index
    cdef float distance
    cdef long int clustersize
    cdef list point_mean
    cdef list prototype_mean
    is_tuple = False
    is_seq = False
    
    if isinstance(point[0], tuple):
        is_tuple = True
    if len(point) > 2:
        is_seq = True
    
    
    
    if is_tuple:
        tupp = []
        # compute mean of each dimension
        grouped_features = zip(*point)
        for xa in grouped_features:
            tupp.append(statistics.mean(xa))
        point_mean = [tuple(tupp)]
    elif is_seq:
        point_mean = [statistics.mean(point)]
    else:
        point_mean = point
    
    for cidx, cluster in enumerate(prototypes): # for each cluster 'cluster'
        distance = 0
        clustersize = len(cluster)
        distances = [0.0]*clustersize
        for idx, prototype in enumerate(cluster): # each 'prototype'
            if is_tuple:
                tupp = []
                # compute mean of each dimension
                grouped_features = zip(*prototype)
                for xa in grouped_features:
                    tupp.append(statistics.mean(xa))
                prototype_mean = [tuple(tupp)]
            elif is_seq:
                prototype_mean = [statistics.mean(prototype)]
            else:
                prototype_mean = prototype 

            distances[idx] = cdist(np.array(prototype_mean).reshape(1, -1),np.array(point_mean).reshape(1, -1), 'euclidean')[0][0]
            GLOBALS.count_dist += 1
        distance = sum(distances)
        distance = distance / float(clustersize) # average distance from all protots
        if minimum_distance > distance:
            minimum_distance = distance
            minimum_idx = cidx
            cp_idx = np.argmin(distances)
            fp_idx = np.argmax(distances)
    
    assigned_clusters.append(minimum_idx)
    #pvotes[minimum_idx][cp_idx] += 1
    #pvotes[minimum_idx] = [x*0.90 for xid, x in enumerate(pvotes[minimum_idx])]
    pvotes[minimum_idx] = [x+1 if xid == cp_idx else x*0.90 for xid, x in enumerate(pvotes[minimum_idx])] # all items in this cluster receive a penalty. Do we penalize other clusters too?
    GLOBALS.CLOSEST[minimum_idx] = cp_idx
    GLOBALS.FARTHEST[minimum_idx] = fp_idx
    
    return (minimum_idx, assigned_clusters, pvotes)
    
    
    
# IMPL3: Baseline with exact stat distance calculation
def st_exact_assign(prototypes, point, assigned_clusters, proto_dist, pvotes, _):
    cdef float minimum_distance = float("inf")
    cdef long long int minimum_idx = -1
    cdef long int cp_idx = -1 # closest proto index
    cdef float distance
    cdef long int clustersize
    
    for cidx, cluster in enumerate(prototypes): # for each cluster 'cluster'
        distance = 0
        clustersize = len(cluster)
        distances = [0.0]*clustersize
        for idx, prototype in enumerate(cluster): # each 'prototype' 
            distances[idx] = euclidean_distances(np.array(prototype).reshape(1, -1), np.array(point).reshape(1, -1))[0][0]
            GLOBALS.count_dist += 1
        distance = sum(distances)
        distance = distance / float(clustersize) # average distance from all protots
        if minimum_distance > distance:
            minimum_distance = distance
            minimum_idx = cidx
            cp_idx = np.argmin(distances)
            fp_idx = np.argmax(distances)
    
    assigned_clusters.append(minimum_idx)
    #pvotes[minimum_idx][cp_idx] += 1
    #pvotes[minimum_idx] = [x*0.90 for xid, x in enumerate(pvotes[minimum_idx])]
    pvotes[minimum_idx] = [x+1 if xid == cp_idx else x*0.90 for xid, x in enumerate(pvotes[minimum_idx])] # all items in this cluster receive a penalty. Do we penalize other clusters too?
    GLOBALS.CLOSEST[minimum_idx] = cp_idx
    GLOBALS.FARTHEST[minimum_idx] = fp_idx
    
    return (minimum_idx, assigned_clusters, pvotes)
    
    
# IMPL2: Baseline with exact distance calculation (voting enabled)
def exact_assign(prototypes, point, assigned_clusters, proto_dist, pvotes, _):
    
    cdef float minimum_distance = float("inf")
    cdef long long int minimum_idx = -1
    cdef long int cp_idx = -1 # closest proto index
    #cdef long int fp_idx = -1 # farthest proto index
    cdef float distance = 0.0
    cdef long int clustersize = 0
    
    for cidx, cluster in enumerate(prototypes): # for each cluster 'cluster'
        distance = 0.0
        clustersize = len(cluster)
        distances = [0.0]*clustersize
        for idx, prototype in enumerate(cluster): # each 'prototype' 
            distances[idx] = dtw(prototype, point, dist_method="euclidean").distance
            GLOBALS.count_dist += 1
        distance = sum(distances)
        distance = distance / float(clustersize) # average distance from all protots
        

        if minimum_distance > distance:
            minimum_distance = distance
            minimum_idx = cidx
            cp_idx = np.argmin(distances)
            #fp_idx = np.argmax(distances)

    assigned_clusters.append(minimum_idx)
    pvotes[minimum_idx] = [x+1 if xid == cp_idx else x*0.90 for xid, x in enumerate(pvotes[minimum_idx])] # all items in this cluster receive a penalty. Do we penalize other clusters too?
    GLOBALS.CLOSEST[minimum_idx] = cp_idx
    #GLOBALS.FARTHEST[minimum_idx] = fp_idx
    
    return (minimum_idx, assigned_clusters, pvotes)
