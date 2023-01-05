
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
#from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean, cdist
from dtw import dtw
import sys, os, re, glob, math
import scipy
import GLOBALS 
from helpers import prototype_distance
import statistics


# init variations
## BASELINE 1: Perfect initialization
def perfect_init(points, labs, nprototypes, nclasses, classdict):
    # 2D points. A jumping window. Initializes the first prototypes of three clusters.

    representative = None

    prototypes = []
    proto_idx = []
    proto_dist = []
    assigned_clusters = [None]*(len(points))
    labs = [classdict[x] for x in labs]
    print(labs)
    for c in range(0,nclasses):
        idxs = [idx for idx,cl in enumerate(labs) if cl == c][0:nprototypes]
        
        pts = [points[x] for x in idxs]
        prototypes.append(pts)
        print(idxs)
        proto_dist.append(prototype_distance(pts))
        proto_idx.append(idxs)
        for idx in idxs:
            assigned_clusters[idx] = c # assign a cluster to idx wala index

    pvotes = [[0]*nprototypes for x in range(nclasses)]
    GLOBALS.NEWEST = {key: -1 for key in range(nclasses)}
    return (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative)


## BASELINE 2: Random initialization
def random_init(batch, _, nprototypes, nclasses, classdict):
    if GLOBALS.DEBUG:
        print('received sequences', len(batch))
        
           
    prototypes = []
    proto_idx = []
    proto_dist = []
    representative = None
    assigned_clusters = [None]*(len(batch))
    
    batch_idx = [i for i in range(len(batch))]
    
    GLOBALS.NEWEST = {key: -1 for key in range(nclasses)}
        
    # pick a random point as first proto
    r = random.choice(batch_idx)
    
    
    prototypes.append(r)

    #print('init proto', r)
    #iterate over batch and pick required protos randomly
    while len(prototypes) < (nclasses*nprototypes):

        r = random.choice(batch_idx)
    
        if r not in prototypes:
            prototypes.append(r)


    
    # if all protos are selected
    proto_idx = []    
    for x in range(nclasses):
        start = x*(nprototypes)
        end = ((x+1)*(nprototypes))
        proto_idx.append(prototypes[start:end])
        for p in prototypes[start:end]:
            assigned_clusters[p] = x
        
        
    prototypes = []
    for c in proto_idx:
        pts = [batch[p] for p in c]
        prototypes.append(pts)
        proto_dist.append([0]*nprototypes)
        
    
    pvotes = [[0]*nprototypes for x in range(nclasses)]

        
 
    return (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative)


## IMPL 4: K-med++ initialization
def nonuniform_init(batch, __, nprototypes, nclasses, classdict):


    proto_dist = []
    (prototypes, proto_idx, representative, assigned_clusters) = Kplusplus(batch, nprototypes, nclasses)
    pvotes = [[0]*nprototypes for x in range(nclasses)]
    GLOBALS.NEWEST = {key: -1 for key in range(nclasses)}
    for c in range(nclasses):
    	proto_dist.append([0]*nprototypes)#prototype_distance(prototypes[c]))
  
    return (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative)

def Kplusplus(batch, nprototypes, nclasses):
    
    init_prototypes = []
    idx = []
    sec_protos = []
    selected = set()
    representative = None
    assigned_clusters = [None]*(len(batch))
    
    batch_idx = [i for i in range(len(batch))]
    
    # pick a random point as first proto
    r = random.choice(batch_idx)
    
    init_prototypes.append(r)

    selected.add(r)
    #iterate over batch. 
    
    distance = []
    
    # K-med++ : Choose the next medoid with a certain probability
    for k in range(nclasses): # k = number of classes/clusters O(k)

        dists = []
        # O(b)
        for candidate in batch_idx:
            
            # compute min distance from one of the closest chosen protos
            latest_proto = init_prototypes[-1]
            d = round((dtw(batch[latest_proto], batch[candidate], dist_method="euclidean").distance)**2, 2) 
            GLOBALS.count_dist += 1
            dists.append(d)
            
        distance.append(dists)
            
        
        #print('len of distance ', len(distance))        
        # Pick other protos of this cluster
        dists = distance[-1]
        di = {k: v for k,v in enumerate(dists)}
        di = sorted(di.items(), key=lambda item: item[1])
        others = []
        i = 0
        
        while len(others) < (nprototypes-1) and i < len(di):
            
            nextt = di[i][0] 
            if nextt not in selected:
                others.append(nextt)
                selected.add(nextt)
            i+=1
        sec_protos.append(others)

            
        assert len(others) == (nprototypes-1)
        
        
  
        
        if k == nclasses-1:
            break
        
        # Pick a proto of other cluster
        closest = [[p[candidate] for p in distance] for candidate in batch_idx]
        #print(closest)
        mindists = [min(x) for x in closest]  
        #print('closest', mindists)
        mindists = np.array(mindists)
        probs = mindists/mindists.sum()
        #print('probs', probs)
        cumprobs = probs.cumsum()
        #print('cumprobs', cumprobs)
        r = random.uniform(0,1)
        #print('r', r)
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        init_prototypes.append(i)
        selected.add(i)
    
    # if all protos are selected
    idx = []    
    for i in range(nclasses):
        l = [init_prototypes[i]]
        l.extend(sec_protos[i])
        idx.append(l)
        
        for p in l:
            assigned_clusters[p] = i
        

    init_prototypes = []
    for c in idx:
        init_prototypes.append([batch[p] for p in c])
    return (init_prototypes, idx, representative, assigned_clusters)

##### Stat implementations
## IMPL 5: K-med++ stat initialization
def st_nonuniform_init(batch, __, nprototypes, nclasses, classdict):


    proto_dist = []
    (prototypes, proto_idx, representative, assigned_clusters) = Kplusplus_stat(batch, nprototypes, nclasses)
    pvotes = [[0]*nprototypes for x in range(nclasses)]
    GLOBALS.NEWEST = {key: -1 for key in range(nclasses)}
    
    for c in range(nclasses):
        proto_dist.append(prototype_distance(prototypes[c]))
    
  

    return (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative)

def Kplusplus_stat(batch, nprototypes, nclasses):
    
    init_prototypes = []
    idx = []
    sec_protos = []
    selected = set()
    representative = None
    assigned_clusters = [None]*(len(batch))
    
    batch_idx = [i for i in range(len(batch))]
    
    # pick a random point as first proto
    r = random.choice(batch_idx)
    
    init_prototypes.append(r)

    selected.add(r)
    #iterate over batch. 
    
    distance = []
    
    # K-med++ : Choose the next medoid with a certain probability
    for k in range(nclasses): # k = number of classes/clusters O(k)

        dists = []
        # O(b)
        for candidate in batch_idx:
            # compute min distance from one of the closest chosen protos
            latest_proto = init_prototypes[-1]
            d = round((euclidean_distances(np.array(batch[latest_proto]).reshape(1, -1),np.array(batch[candidate]).reshape(1, -1))[0][0])**2, 2)
            GLOBALS.count_dist += 1
            dists.append(d)
            
        distance.append(dists)
            
        
        #print('len of distance ', len(distance))        
        # Pick other protos of this cluster
        dists = distance[-1]
        di = {k: v for k,v in enumerate(dists)}
        di = sorted(di.items(), key=lambda item: item[1])
        others = []
        i = 0
        
        while len(others) < (nprototypes-1) and i < len(di):
            
            nextt = di[i][0] 
            if nextt not in selected:
                others.append(nextt)
                selected.add(nextt)
            i+=1
        sec_protos.append(others)
        
        
        assert len(others) == (nprototypes-1)
        

        
        if k == nclasses-1:
            break
        
        # Pick a proto of other cluster
        closest = [[p[candidate] for p in distance] for candidate in batch_idx]
        #print(closest)
        mindists = [min(x) for x in closest]  
        #print('closest', mindists)
        mindists = np.array(mindists)
        probs = mindists/mindists.sum()
        #print('probs', probs)
        cumprobs = probs.cumsum()
        #print('cumprobs', cumprobs)
        r = random.uniform(0,1)
        #print('r', r)
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        init_prototypes.append(i)
        selected.add(i)
        
    # if all protos are selected
    idx = []    
    for i in range(nclasses):
        l = [init_prototypes[i]]
        l.extend(sec_protos[i])
        idx.append(l)
        
        for p in l:
            assigned_clusters[p] = i
        
    init_prototypes = []
    for c in idx:
        init_prototypes.append([batch[p] for p in c])
    return (init_prototypes, idx, representative, assigned_clusters)
    

##### Stat implementations
## IMPL 6: K-med++ aggregated initialization
def aggr_nonuniform_init(batch, __, nprototypes, nclasses, classdict):


    proto_dist = []
    (prototypes, proto_idx, representative, assigned_clusters) = Kplusplus_aggr(batch, nprototypes, nclasses)
    pvotes = [[0]*nprototypes for x in range(nclasses)]
    GLOBALS.NEWEST = {key: -1 for key in range(nclasses)}
    
    for c in range(nclasses):
        proto_dist.append([0]*nprototypes)#prototype_distance(prototypes[c]))
    
  

    return (prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative)

def Kplusplus_aggr(batch, nprototypes, nclasses):
    
    init_prototypes = []
    idx = []
    sec_protos = []
    selected = set()
    representative = None
    assigned_clusters = [None]*(len(batch))
    is_tuple = False
    is_seq = False
    batch_idx = [i for i in range(len(batch))]
    
    # check if the data type is tuple or others
    if isinstance(batch[0][0], tuple):
        is_tuple = True
    if len(batch[0]) > 2:
        is_seq = True
    # pick a random point as first proto
    r = random.choice(batch_idx)
    
    init_prototypes.append(r)

    selected.add(r)
    #iterate over batch. 
    
    distance = []
    
    # K-med++ : Choose the next medoid with a certain probability
    for k in range(nclasses): # k = number of classes/clusters O(k)

        dists = []
        batch_latest_proto_mean, batch_candidate_mean = -1, -1
        latest_proto = init_prototypes[-1]
        
        if is_tuple:
            tupp = []
            # compute mean of each dimension
            grouped_features = zip(*batch[latest_proto])
            for xa in grouped_features:
                tupp.append(statistics.mean(xa))
            batch_latest_proto_mean = [tuple(tupp)]
        elif is_seq:
            batch_latest_proto_mean = [statistics.mean(batch[latest_proto])]
        else:
            batch_latest_proto_mean = batch[latest_proto]

        # O(b)
        for candidate in batch_idx:
            # compute min distance from one of the closest chosen protos
            if is_tuple:
                tupp = []
                # compute mean of each dimension
                grouped_features = zip(*batch[candidate])
                for xa in grouped_features:
                    tupp.append(statistics.mean(xa))
                batch_candidate_mean = [tuple(tupp)]
            elif is_seq:
                batch_candidate_mean = [statistics.mean(batch[candidate])]
            else:
                batch_candidate_mean = batch[candidate]            
            d = round((cdist(np.array(batch_latest_proto_mean).reshape(1, -1),np.array(batch_candidate_mean).reshape(1, -1), 'euclidean')[0][0])**2, 2)
            GLOBALS.count_dist += 1
            dists.append(d)
            
        distance.append(dists)
            
        
        #print('len of distance ', len(distance))        
        # Pick other protos of this cluster
        dists = distance[-1]
        di = {k: v for k,v in enumerate(dists)}
        di = sorted(di.items(), key=lambda item: item[1])
        others = []
        i = 0
        
        while len(others) < (nprototypes-1) and i < len(di):
            
            nextt = di[i][0] 
            if nextt not in selected:
                others.append(nextt)
                selected.add(nextt)
            i+=1
        sec_protos.append(others)
        
        
        assert len(others) == (nprototypes-1)
        

        
        if k == nclasses-1:
            break
        
        # Pick a proto of other cluster
        closest = [[p[candidate] for p in distance] for candidate in batch_idx]
        #print(closest)
        mindists = [min(x) for x in closest]  
        #print('closest', mindists)
        mindists = np.array(mindists)
        probs = mindists/mindists.sum()
        #print('probs', probs)
        cumprobs = probs.cumsum()
        #print('cumprobs', cumprobs)
        r = random.uniform(0,1)
        #print('r', r)
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        init_prototypes.append(i)
        selected.add(i)
        
    # if all protos are selected
    idx = []    
    for i in range(nclasses):
        l = [init_prototypes[i]]
        l.extend(sec_protos[i])
        idx.append(l)
        
        for p in l:
            assigned_clusters[p] = i
        
    init_prototypes = []
    for c in idx:
        init_prototypes.append([batch[p] for p in c])
    return (init_prototypes, idx, representative, assigned_clusters)
    
