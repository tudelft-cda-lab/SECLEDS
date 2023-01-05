import numpy as np
from scipy.spatial.distance import cdist
import statistics, math
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_distro(now_str, config_name, dataset, trial, overall_means, overall_stdevs, cluster_means, cluster_stdevs):
    is_tuple = len(overall_means) > 1
    
    
    print(overall_means, overall_stdevs, cluster_means, cluster_stdevs)
    for i in range(len(overall_means)): 
        fig = plt.figure(figsize=(10,5))
        
        plt.title('Feature distribution for feature '+str(i))
        
        r1 = overall_means[i]-5*overall_stdevs[i]
        r2 = overall_means[i]+5*overall_stdevs[i]
        x = np.arange(r1, r2, 0.01)
        plt.plot(x, norm.pdf(x, overall_means[i], overall_stdevs[i]), label='overall')
    
        for c in range(len(cluster_means)):
            if is_tuple:
                r1 = cluster_means[c][i]-5*cluster_stdevs[c][i]
                r2 = cluster_means[c][i]+5*cluster_stdevs[c][i]
                x = np.arange(r1, r2, 0.01)
                plt.plot(x, norm.pdf(x, cluster_means[c][i], cluster_stdevs[c][i]), label='clus'+str(c))
            else:
                r1 = cluster_means[c]-5*cluster_stdevs[c]
                r2 = cluster_means[c]+5*cluster_stdevs[c]
                x = np.arange(r1, r2, 0.01)
                plt.plot(x, norm.pdf(x, cluster_means[c], cluster_stdevs[c]), label='clus'+str(c))
        plt.legend()
        plt.savefig(now_str+'/'+'explain-'+config_name+'-'+dataset+'-trial'+str(trial)+'-feature'+str(i)+'.png')
        plt.close(fig)    

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
        grouped_features = zip(*point)
        tupp = []
        for xa in grouped_features:
            tupp.append(statistics.mean(xa))
        point_aggr = tupp
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
                tupp = []
                grouped_features = zip(*prototype)
                for xa in grouped_features:
                    tupp.append(statistics.mean(xa))
                cluster_aggr.append(tuple(tupp))
            elif is_seq:
                cluster_aggr.append(statistics.mean(prototype))
            else:
                cluster_aggr.append((prototype[0], prototype[1]))

        # Compute mean and stdev of each prototype
        n = len(prototypes)
        if isinstance(cluster_aggr[0], tuple):
            tupp_m, tupp_s = [], []
            grouped_features = zip(*cluster_aggr)
            for f1 in grouped_features:
                tupp_m.append(statistics.mean(f1))
                tupp_s.append(statistics.stdev(f1)) 
            means.append(tuple(tupp_m))
            stds.append(tuple(tupp_s))
        else:
            f1 = cluster_aggr
            means.append(statistics.mean(f1))
            stds.append(statistics.stdev(f1))
    
    return (means, stds)