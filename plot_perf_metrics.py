import os, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def save_file(df_, prefix, feat, now_str):
    fig = plt.figure(figsize=(10,10))
    name = prefix+'-'+feat
    plt.title(name)
    
    dd=pd.melt(df_,id_vars=['Group'],value_vars=[feat],var_name='metrics')
    ax = sns.boxplot(x='Group',y='value',data=dd,hue='metrics')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        
    plt.savefig(now_str+'/'+name+'.png', dpi=600)
    
    
def plot_perf_metrics(fname, now_str):
    with open(fname, 'r') as f:
        content = json.load(f)
        f1= []
        precision = []
        recall = []
        tp, tn, fp, fn = [], [], [], []
        corrects, mistakes = [], []
        purity = []
        complete = []
        init_purity= []
        init_complete = []
        config = []
        dists = []
        runtime = []
        runs = None
        dataset = None
        options = ['perfect-naive', 'perfect-seqclu',  'rand-naive', 'rand-seqclu', 'K++-naive', 'K++-seqclu']
        for pid, perf in enumerate(content):
        
            if len(perf.keys()) == 0:
                continue
                
                
            
            
            #print(con)   
            #print(perf['metrics'])
            #print('-----------')
            
            dataset = perf['dataset']
            mmetrics = perf['metrics']
            con = ' | '.join(perf['config'].values()) # perf['data_properties']['nprototypes']#
            nprototypes = perf['data_properties']['nprototypes']
            nclasses = perf['data_properties']['nclasses']
            print(con, nprototypes, mmetrics)
            
            init_purity.extend(mmetrics['init'])
            init_complete.extend(mmetrics['init_c'])
            purity.extend(mmetrics['purity'])
            complete.extend(mmetrics['complete'])
            f1.extend(mmetrics['f1'])
            recall.extend(mmetrics['recall'])
            precision.extend(mmetrics['precision'])
            tp.extend(mmetrics['TP'])
            tn.extend(mmetrics['TN'])
            fp.extend(mmetrics['FP'])
            fn.extend(mmetrics['FN'])
            
            c = [x+y for x,y in zip(mmetrics['TP'],mmetrics['TN'])]
            m = [x+y for x,y in zip(mmetrics['FP'],mmetrics['FN'])]
            tot = [x+y for x,y in zip(c,m)]
            corrects.extend([x/y for x,y in zip(c,tot)])
            mistakes.extend([x/y for x,y in zip(m,tot)])
            
            dists.extend(mmetrics['dists'])
            runtime.extend(mmetrics['runtime'])
            
            runs = mmetrics['runs']
            
            config.extend(['|'.join([x[:4] for x in perf['config'].values()])]*runs)
            #config.extend([options[pid]]*runs)
            #config.extend([str(nprototypes)]*runs)
            
        #print(init_purity)
        
        #print(config)

        df = pd.DataFrame(
        {'Group':config,\
        'init_purity':init_purity,
        'init_complete':init_complete,
        'purity':purity, 
        'complete':complete,
        'f1':f1,
        'precision':precision,
        'recall': recall,
        'tp': tp, 
        'tn': tn, 
        'fp': fp, 
        'fn': fn, 
        'corrects':corrects,
        'mistakes':mistakes,
        'dists': dists,
        'runtime':runtime
        }
        )
        
        prefix = dataset+'-TRIALS-'+str(runs)+'-NCLASS-'+str(nclasses)+'-NPROTO-'+str(nprototypes)
        name = now_str+'/'+prefix+'-avg-scores'
        outfile = open(name, 'w')
        cons = set(df['Group'])
        for cid, conf in enumerate(cons):
            outfile.write('\n\n::: %s :::\n'%(conf))
            for column in df:
                if column == 'Group':
                    continue
            
                score = np.average(df.loc[df['Group'] == conf, column])
                outfile.write('\t'+column+'= '+str(score)+'\n')
                if cid == 0:    
                    save_file(df[['Group',column]], prefix, column, now_str)
        outfile.close()
        
        

        
        




### structure
'''perf_metrics = {
                'init_purity_mean' : np.mean(init_purity),\
                'init_purity_std' :        np.std(init_purity), \
                'purity_mean' : np.mean(purity),\
                'purity_std' :  np.std(purity), \
                'precision_mean' : np.mean(precision),\
                'precision_std' :  np.std(precision), \
                'recall_mean' : np.mean(recall),\
                'recall_std' :  np.std(recall), \
                'f1_mean' : np.mean(f1),\
                'f1_std' :  np.std(f1), \
                'dists_mean' : np.mean(dists),\
                'dists_std' :  np.std(dists), \
                'runtime_mean' :  np.mean(time_to_cluster),\
                'runtime_std' :   np.std(time_to_cluster), \
                'runs' : TRIALS
                }
        
    perf = {}
    
    perf['dataset'] = str(DATASET)
    perf['config'] = {
        'INIT': str(INIT.__name__) ,
        'ASSIGN': str(ASSIGN.__name__),
        'UPDATE': str(UPDATE.__name__)
    }
    perf['cluster_properties'] = {
        'batchsize': batchsize,
        'clustersize': dict(Counter(assigned_clusters).items())
    }
    
    perf['data_properties'] = {
        'nsequences': len(X),
        'nclasses': nclasses,
        'nprototypes': nprototypes,
        'classes': dict(classdict.items()),
        'class_distro': dict(Counter(labs).items())
    }
    
    perf['metrics'] = perf_metrics'''
    
    
#plot_perf_metrics('031221-223215-plots/exp-results.txt', '031221-223215-plots')
