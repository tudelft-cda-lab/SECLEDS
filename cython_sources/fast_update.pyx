#!python
#cython: language_level=3

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from dtw import dtw
import GLOBALS 
import statistics
from scipy.spatial.distance import cdist


# IMPL1: SeqClu with proto voting. replace the lowest value proto
def prototypevoting_update(prototypes, point, minimum_idx, proto_idx, pidx, proto_dist, pvotes, representative, _buffer=None):
    cdef int replaced, least_value_idx
    cdef long int least_votes
    cdef float dist
    
    replaced = 0
    allproto =  [item for sublist in proto_idx for item in sublist]
    if pidx in allproto: # only consider if its a new proto (a proto can only be in a single cluster and only 1 occurance at a time)
        return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)
    
    ## Replace the prototype with least value and distribute the votes
    least_dists = [x[0] for x in sorted(enumerate(pvotes[minimum_idx]), key=lambda tup: tup[1])]
    least_value_idx = least_dists[0]
    if GLOBALS.NEWEST[minimum_idx] == least_value_idx:
        least_value_idx = least_dists[1]
    
    least_value_proto = prototypes[minimum_idx][least_value_idx]
    least_votes = pvotes[minimum_idx][least_value_idx]
    # replace first
    prototypes[minimum_idx][least_value_idx] = point # replace the least valued proto of the closest cluster
    proto_idx[minimum_idx][least_value_idx] = pidx
    pvotes[minimum_idx][least_value_idx] = 0 # new proto has no votes yet
    GLOBALS.NEWEST[minimum_idx] = least_value_idx
    replaced = 1
    if least_votes > 0:
        # then we distribute votes
        distances = [0.0]*len(prototypes[minimum_idx])
        for idx, proto in enumerate(prototypes[minimum_idx]):
            dist = dtw(proto, least_value_proto, dist_method="euclidean").distance
            #GLOBALS.count_dist += 1
            dist = 1/(dist) if dist > 0.0 else 0.0
            distances[idx] = dist # Inverse of distance is similarity
        tdist = sum(distances)
        if tdist > 0:
            frac_votes = [round(round(x/float(tdist),1)*least_votes,1) for x in distances]
            pvotes[minimum_idx] = [round(x+frac_votes[i],1) for i,x in enumerate(pvotes[minimum_idx])] 

    return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)
    
# IMPL3: SeqClu with proto voting (stat version). replace the lowest value proto
def st_prototypevoting_update(prototypes, point, minimum_idx, proto_idx, pidx, proto_dist, pvotes, representative, _buffer=None):
    cdef int replaced, least_value_idx
    cdef long int least_votes
    cdef float dist
    
    replaced = 0
    allproto =  [item for sublist in proto_idx for item in sublist]
    if pidx in allproto: # only consider if its a new proto (a proto can only be in a single cluster and only 1 occurance at a time)
        return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)
    
    ## Replace the prototype with least value and distribute the votes
   
    least_dists = [x[0] for x in sorted(enumerate(pvotes[minimum_idx]), key=lambda tup: tup[1])]
    least_value_idx = least_dists[0]
    if GLOBALS.NEWEST[minimum_idx] == least_value_idx:
        least_value_idx = least_dists[1]
    
    least_value_proto = prototypes[minimum_idx][least_value_idx]
    least_votes = pvotes[minimum_idx][least_value_idx]
    # replace first
    prototypes[minimum_idx][least_value_idx] = point # replace the least valued proto of the closest cluster
    proto_idx[minimum_idx][least_value_idx] = pidx
    pvotes[minimum_idx][least_value_idx] = 0 # new proto has no votes yet
    GLOBALS.NEWEST[minimum_idx] = least_value_idx
    replaced = 1
    if least_votes > 0:
        # then we distribute votes
        distances = [0.0]*len(prototypes[minimum_idx])
        for idx, proto in enumerate(prototypes[minimum_idx]):
            dist = euclidean_distances(np.array(proto).reshape(1, -1),np.array(least_value_proto).reshape(1, -1))[0][0]
            #GLOBALS.count_dist += 1
            dist = 1/(dist) if dist > 0.0 else 0.0
            distances[idx] = dist # Inverse of distance is similarity
        
        tdist = sum(distances)
        if tdist > 0:
            frac_votes = [round(round(x/float(tdist),1)*least_votes,1) for x in distances]
            pvotes[minimum_idx] = [round(x+frac_votes[i],1) for i,x in enumerate(pvotes[minimum_idx])] 

    return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)


# IMPL4: SeqClu with proto voting (aggregated version). replace the lowest value proto
def aggr_prototypevoting_update(prototypes, point, minimum_idx, proto_idx, pidx, proto_dist, pvotes, representative, _buffer=None):
    cdef int replaced, least_value_idx
    cdef long int least_votes
    cdef float dist
    
    replaced = 0
    allproto =  [item for sublist in proto_idx for item in sublist]
    if pidx in allproto: # only consider if its a new proto (a proto can only be in a single cluster and only 1 occurance at a time)
        return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)
    cdef list least_value_proto_mean
    cdef list proto_mean
    is_tuple = False
    is_seq = False
    
    if isinstance(point[0], tuple):
        is_tuple = True
    if len(point) > 2:
        is_seq = True
    ## Replace the prototype with least value and distribute the votes
   
    least_dists = [x[0] for x in sorted(enumerate(pvotes[minimum_idx]), key=lambda tup: tup[1])]
    least_value_idx = least_dists[0]
    if GLOBALS.NEWEST[minimum_idx] == least_value_idx:
        least_value_idx = least_dists[1]
    
    least_value_proto = prototypes[minimum_idx][least_value_idx]
    least_votes = pvotes[minimum_idx][least_value_idx]
    # replace first
    prototypes[minimum_idx][least_value_idx] = point # replace the least valued proto of the closest cluster
    proto_idx[minimum_idx][least_value_idx] = pidx
    pvotes[minimum_idx][least_value_idx] = 0 # new proto has no votes yet
    GLOBALS.NEWEST[minimum_idx] = least_value_idx
    replaced = 1
    if least_votes > 0:
        
        
        if is_tuple:
            # compute mean of each dimension
            xa, xb = zip(*least_value_proto)
            least_value_proto_mean = [(statistics.mean(xa), statistics.mean(xb))]
        elif is_seq:
            least_value_proto_mean = [statistics.mean(least_value_proto)]
        else:
            least_value_proto_mean = least_value_proto
        
        # then we distribute votes
        distances = [0.0]*len(prototypes[minimum_idx])
        for idx, proto in enumerate(prototypes[minimum_idx]):
            if is_tuple:
                # compute mean of each dimension
                xa, xb = zip(*proto)
                proto_mean = [(statistics.mean(xa), statistics.mean(xb))]
            elif is_seq:
                proto_mean = [statistics.mean(proto)]
            else:
                proto_mean = proto 

            dist = cdist(np.array(proto_mean).reshape(1, -1),np.array(least_value_proto_mean).reshape(1, -1), 'euclidean')[0][0]
            #GLOBALS.count_dist += 1
            dist = 1/(dist) if dist > 0.0 else 0.0
            distances[idx] = dist # Inverse of distance is similarity
        
        tdist = sum(distances)
        if tdist > 0:
            frac_votes = [round(round(x/float(tdist),1)*least_votes,1) for x in distances]
            pvotes[minimum_idx] = [round(x+frac_votes[i],1) for i,x in enumerate(pvotes[minimum_idx])] 

    return (proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, replaced)
