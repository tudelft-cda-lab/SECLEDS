

import sys, os, json
import numpy as np


fname = sys.argv[1]

with open(fname, 'r') as f:
    content = json.load(f)
    for pid, perf in enumerate(content):
        if pid == 0:
            print('---dataset: ', perf['dataset'])
            print('--- num samples: ', perf['data_properties']['nsequences'], ' of dimesnions ', perf['data_properties']['dim'], '  DRIFT: ', perf['data_properties']['drift'][0])
            print('---nclasses, nprotos: ', perf['data_properties']['nclasses'], perf['data_properties']['nprototypes'])
        configname = perf['config']
        print('\t---', configname)
        if 'banditPAM' in configname:
            print('\t\t---Overall F1:', perf['metrics']['f1'][0])
            print('\t\t---F1_err:', 0)
            print('\t\t---Overall Purity:', perf['metrics']['purity'][0])
            print('\t\t---Proto Purity:', perf['metrics']['proto_purity'][0], ' | Clusters dicovered by protos:', perf['metrics']['clusters_discovered'][0])
            print('\t\t---TTC:', perf['metrics']['runtime'][0])   
            print('\t\t---TTC_err:', np.std(perf['metrics']['runtime'])/np.sqrt(perf['metrics']['runs']))     
        else:
            print('\t\t---Overall F1:', np.mean(perf['metrics']['f1']))
            print('\t\t---F1_err:', np.std(perf['metrics']['f1'])/np.sqrt(perf['metrics']['runs']))
            print('\t\t---Overall Purity:', np.mean(perf['metrics']['purity']))
            print('\t\t---Proto Purity:', np.mean(perf['metrics']['proto_purity']), ' | Clusters dicovered by protos:', np.mean(perf['metrics']['clusters_discovered']))
            print('\t\t---TTC:', np.mean(perf['metrics']['runtime']))
            print('\t\t---TTC_err:', np.std(perf['metrics']['runtime'])/np.sqrt(perf['metrics']['runs']))
