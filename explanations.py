import numpy as np
from scipy.spatial.distance import cdist
import statistics, math


def explain_all_data(n_so_far, means_old, stds_old, point):
    is_tuple = False
    is_seq = False
    
    if isinstance(point[0], tuple):
        is_tuple = True
    if len(point) > 2:
        is_seq = True
        
    point_aggr = None
    # First compute aggregate of the point object if it is a sequence or multi-dim sequence
    if is_tuple:
        # compute mean of each dimension
        xa, xb = zip(*point)
        point_aggr = [statistics.mean(xa), statistics.mean(xb)]
    elif is_seq:
        point_aggr = [statistics.mean(point)]
    else:
        point_aggr = point
                
                
    n = n_so_far + 1
    
    means_new, stds_new = [] , []
    
    # Then compute mean and std dev of each feature
    for did, dim in enumerate(point_aggr):
        mean_new = ((means_old[did]*n_so_far) + dim)/float(n)
        std_new = math.sqrt(((n-2)*stds_old[did]**2 + ((n-1)*(mean_new - means_old[did])**2) + (float(dim)-mean_new)**2)/n_so_far)
        
        means_new.append(mean_new)
        stds_new.append(std_new)

    return (n, means_new, stds_new)
    
    
    
    
def explain_cluster(prototypes):
    
    means, stds = [], []
    
    is_tuple = False
    is_seq = False
    
    if isinstance(prototypes[0][0][0], tuple):
        is_tuple = True
    if len(prototypes[0][0]) > 2:
        is_seq = True
        
    for cluster in prototypes:
        cluster_aggr = []
        # First compute aggregate of the prototype object if it is a sequence or multi-dim sequence
        for prototype in cluster:
            if is_tuple:
                # compute mean of each dimension
                xa, xb = zip(*prototype)
                cluster_aggr.append((statistics.mean(xa), statistics.mean(xb)))
            elif is_seq:
                cluster_aggr.append(statistics.mean(prototype))
            else:
                cluster_aggr.append((prototype[0], prototype[1]))

        # Compute mean and stdev of each prototype
        n = len(prototypes)
        if isinstance(cluster_aggr[0], tuple):
            f1,f2 = zip(*cluster_aggr)
            means.append((statistics.mean(f1), statistics.mean(f2)))
            stds.append((statistics.stdev(f1), statistics.mean(f2)))
        else:
            f1 = cluster_aggr
            means.append(statistics.mean(f1))
            stds.append(statistics.stdev(f1))
    
    return (means, stds)