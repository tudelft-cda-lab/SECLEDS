import sys, os, re, glob, math
import seaborn as sns
from datetime import datetime
from data_generation import generate_circles, generate_blobs, generate_curves
from helpers import plot_data
import GLOBALS 





if len(sys.argv) < 3:
    print('USAGE: create_data.py {sine-curve|blobs|circles} {#clusters} {#samples}')
    sys.exit()
DATASET = sys.argv[1]
nclasses = int(sys.argv[2])
NEW_SAMPLES = int(sys.argv[3])
pal = sns.color_palette("hls", nclasses)
now = datetime.now()
now_str = str(NEW_SAMPLES)+'samples,'+str(nclasses)+'classes-'+now.strftime("%d%m%y-%H%M%S")+'-plots'
GLOBALS.init()

print('Getting the data...')
if DATASET == 'sine-curve':
    from sklearn.manifold import TSNE
    (X, labs, classdict) = generate_curves(NEW_SAMPLES, nclasses, 100, now_str)
    X_embedded = TSNE(random_state=42, n_components =2).fit_transform(X)
elif DATASET == 'blobs':
    (X, labs, classdict) = generate_blobs(NEW_SAMPLES, nclasses,  now_str)
    X_embedded = X
elif DATASET == 'circles':
    (X, labs, classdict) = generate_circles(NEW_SAMPLES, nclasses,  now_str)
    X_embedded = X
else:
    print('Dataset not recognized')
    sys.exit(-1) 

print('Plotting raw data...')
plot_data(X_embedded, labs, classdict, pal, DATASET, 'datasets/'+DATASET+'/'+now_str[:-6])
