#!python
#cython: language_level=3

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from dtw import dtw
import GLOBALS 

# IMPL3: Baseline with exact stat distance calculation
def st_exact_assign(prototypes, point, assigned_clusters, proto_dist, pvotes, _):
    cdef float minimum_distance = 10000000
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
    
    cdef float minimum_distance = 10000000
    cdef long long int minimum_idx = -1
    cdef long int cp_idx = -1 # closest proto index
    cdef float distance
    cdef long int clustersize
    
    for cidx, cluster in enumerate(prototypes): # for each cluster 'cluster'
        distance = 0
        clustersize = len(cluster)
        distances = [0.0]*clustersize
        for idx, prototype in enumerate(cluster): # each 'prototype' 
            distances[idx] = dtw(prototype, point, distance_only=True, dist_method="euclidean").distance
            GLOBALS.count_dist += 1
        distance = sum(distances)
        distance = distance / float(clustersize) # average distance from all protots
        

        if minimum_distance > distance:
            minimum_distance = distance
            minimum_idx = cidx
            cp_idx = np.argmin(distances)
            fp_idx = np.argmax(distances)

    assigned_clusters.append(minimum_idx)
    pvotes[minimum_idx] = [x+1 if xid == cp_idx else x*0.90 for xid, x in enumerate(pvotes[minimum_idx])] # all items in this cluster receive a penalty. Do we penalize other clusters too?
    GLOBALS.CLOSEST[minimum_idx] = cp_idx
    GLOBALS.FARTHEST[minimum_idx] = fp_idx
    
    return (minimum_idx, assigned_clusters, pvotes)
