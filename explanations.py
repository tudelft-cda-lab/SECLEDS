import numpy as np
from scipy.spatial.distance import cdist
import statistics, math
from scipy.stats import norm
import matplotlib.pyplot as plt
from helpers import prototype_distance

def plot_distro(now_str, config_name, dataset, trial, overall_means, overall_stdevs, cluster_means, cluster_stdevs, central_protos):
    is_tuple = len(overall_means) > 1
    is_seq = len(central_protos[0]) > 2
    for i in range(len(overall_means)):
        fig = plt.figure(figsize=(10,5))
        plt.title('Feature distribution for feature '+str(i))

        r1 = overall_means[i]-5*overall_stdevs[i]
        r2 = overall_means[i]+5*overall_stdevs[i]
        x = np.arange(r1, r2, 0.01)
        plt.plot(x, norm.pdf(x, overall_means[i], overall_stdevs[i]), c='black', label='overall')

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
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(now_str+'/'+'featuredistro-'+config_name+'-'+dataset+'-trial'+str(trial)+'-feature'+str(i)+'.png')
        plt.close(fig)

    for c in range(len(central_protos)):
        this_cluster_means = cluster_means[c]
        this_cluster_stdev = cluster_stdevs[c]


        fig = plt.figure(figsize=(20, 20))
        plt.suptitle('Explanation for cluster '+str(c))

        if is_tuple:
            # run loop for each feature
            numplots = len(overall_means)

            for i in range(numplots):  # for each feature
                ax = fig.add_subplot(numplots, 2, (i*2)+1)
                ax.title.set_text('Feature distribution compared to all')
                r1 = this_cluster_means[i] - 5 * this_cluster_stdev[i]
                r2 = this_cluster_means[i] + 5 * this_cluster_stdev[i]
                x = np.arange(r1, r2, 0.01)
                ax.plot(x, norm.pdf(x, this_cluster_means[i], this_cluster_stdev[i]), c="red", label='Feature'+str(i))

                # Plot overall distro
                r1 = overall_means[i] - 5 * overall_stdevs[i]
                r2 = overall_means[i] + 5 * overall_stdevs[i]
                x = np.arange(r1, r2, 0.01)
                ax.plot(x, norm.pdf(x, overall_means[0], overall_stdevs[0]), c="black", label='Overall data')
                ax.legend(loc='upper right')
                ax.grid(True)
                # Plot central proto
                ax = fig.add_subplot(numplots, 2, (i*2)+2)
                ax.title.set_text('Central prototype sequence, Feature '+str(i))
                if dataset == 'multi-chars':
                    features = list(zip(*(central_protos[c])))
                    ax.plot(features[0], features[1])
                elif is_seq:
                    features = list(zip(*(central_protos[c])))
                    ax.plot(range(len(features[i])), features[i], label='Feature '+str(i))
                else:
                    ax.scatter(central_protos[c][0], central_protos[c][1])
                #ax.set(ylabel="Feat "+ str(i))
                ax.grid(True)
        else:
            # Plot distro for cluster
            ax = fig.add_subplot(1, 2, 1)
            ax.title.set_text('Feature distribution compared to all')
            r1 = this_cluster_means - 5 * this_cluster_stdev
            r2 = this_cluster_means + 5 * this_cluster_stdev
            x = np.arange(r1, r2, 0.01)
            ax.plot(x, norm.pdf(x, this_cluster_means, this_cluster_stdev), c="red",label='Feature 0')

            # Plot overall distro
            r1 = overall_means[0] - 5 * overall_stdevs[0]
            r2 = overall_means[0] + 5 * overall_stdevs[0]
            x = np.arange(r1, r2, 0.01)
            ax.plot(x, norm.pdf(x, overall_means[0], overall_stdevs[0]), c="black", label='Overall data')
            ax.legend(loc='upper right')
            ax.grid(True)
            # Plot central proto
            ax = fig.add_subplot(1, 2, 2)
            ax.title.set_text('Central prototype sequence')
            ax.plot(range(len(central_protos[c])), central_protos[c], label='Feature')
            #ax.set(ylabel="Feat 0")
            ax.grid(True)
        plt.savefig(now_str+'/'+'explain-'+config_name+'-'+dataset+'-trial'+str(trial)+'-cluster'+str(c)+'.png')
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
    
    means, stds, central_protos = [], [], []


    is_tuple = False
    is_seq = False
    
    if isinstance(prototypes[0][0][0], tuple):
        is_tuple = True
    if len(prototypes[0][0]) > 2:
        is_seq = True
        
    for cid, cluster in enumerate(prototypes):
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
        # Find the prototype with the least average distance to all other prototypes
        dist_matrix = prototype_distance(cluster, seq=True)
        relative_dists = {}
        for pid in range(len(cluster)):
            relative_dist = [dist for tupp,dist in dist_matrix.items() if tupp[0] == pid]
            relative_dist = sum(relative_dist)/float(len(cluster))
            relative_dists[pid] = relative_dist
        central_proto_idx = min(relative_dists, key=relative_dists.get)
        central_proto = cluster[central_proto_idx]
        central_protos.append(central_proto)
    return (means, stds, central_protos)