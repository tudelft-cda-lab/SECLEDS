import asyncio, aiofiles

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE
import sys, os, math, copy
import time, json, csv
from IPython import display
from collections import Counter
import itertools, configparser, argparse

# Importing source code
from cython_sources.fast_assign import exact_assign
from cython_sources.fast_update import prototypevoting_update
from init_routines import nonuniform_init

from helpers import format_scores, plot_votes, plot_over_time, shuffle_stream, plot_proto, plot_seq, plot_data, plot_all, plot_onlineBL, plot_offlineBL, plot_heatmap, plot_medoids, add_drift, plot_letters, prototype_distance
from cython_sources.eval_pairwise import adjacency_accuracy, evaluate_purity_complete
from evaluations import evaluate_PR,  proto_purity

import GLOBALS



async def streamfunc(filename, nclasses, nprototypes, batchsize, _len=50):
	# Note that this script does not ensure that the periodic storing of medoids is unique!
	buffer = {}
	buffer_new = {}
	data = []
	fig = plt.figure(figsize=(10, 10))
	classdict = dict(zip(list(range(nclasses)), list(range(nclasses))))
	classes = list(classdict.values())
	lid = False
	init, run_clus = True, False
	init_count = 0
	overallcount = 0
	checkcount = 0
	seqcount = 0
	savecounter = 1
	prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative = [], [], [], [], [], None
	timer_start = time.time()
	async with aiofiles.open(filename, mode='r') as f:
		async for line in f:
			if not lid:
				lid = True
				continue
			netflow = line.split(',')
			
			# parse the line
			ts = netflow[0]
			bytes = int(netflow[1])
			pkts = int(netflow[2])
			srcip = netflow[5]
			dstip = netflow[3]
			srcport = int(netflow[8])
			dstport = int(netflow[7])
			label = netflow[9]

			features = (pkts,bytes,srcport,dstport)
			print('.', end=' ', flush=True)

			pair = (srcip)
			# get source and dest IPs
			# Check if there's already an entry in the buffer
			if pair not in buffer.keys():
				buffer[pair] = []
			# If not, then add entry to the buffer
			
			# Add the netflow to the buffer
			buffer[pair].append(features)
			overallcount += 1

			# Check if length == len. 
			if len(buffer[pair]) >= _len:
				init_count = len([k for k,v in buffer.items() if len(v) >= _len])
				if init_count == batchsize and init:
					# If yes, update the counter and see if requirement to start init is complete
					# If yes, run the init and set run_algo to true, set all buffers to empty again
					# create new buffer with only the ones that have enough length
					keys_new = [k for (k,v) in buffer.items() if len(v) >= _len ]
					#print(keys_new, [len(buffer[x]) for x in keys_new])
					for k in keys_new:
						b = len(buffer[k])
						buffer_new[k] = buffer[k][0:_len]
						buffer[k] = buffer[k][_len:]
						
						data.append(buffer_new[k])
						
						checkcount += len(buffer_new[k])
						assert b == len(buffer_new[k])+len(buffer[k])
						if len(buffer[k]) == 0:
							del buffer[k]
					#print('Initializing', buffer_new.keys(), [len(x) for x in buffer_new.values()])
					print('I', end=' ', flush=True)
					stream = [x for x in buffer_new.values()]
					(prototypes, proto_idx, assigned_clusters, proto_dist, pvotes, representative) = nonuniform_init(stream,[],nprototypes, nclasses,classdict)
					
					run_clus = True
					init = False
					
					# Plot protos
					if RT_ANIMATION:
						ann_new = [x for x, y in enumerate(data)]
						plot_proto(fig, 'SECLEDS', assigned_clusters, proto_idx, data, ann_new, pal, classes)
						plt.show(block=False)
						plt.pause(1.0)
						plt.close()
					seqcount = init_count
					continue
				
				if run_clus:
					seqcount += 1
					seq = buffer[pair][0:_len]
					#print('Clustering', pair, len(seq), len(seq[0]))
						
					print('C', end=' ', flush=True)
					data.append(seq)
					# Cluster assignment phase: assigning cluster to a point
					(minimum_idx, assigned_clusters, pvotes) = exact_assign(prototypes, seq, assigned_clusters, proto_dist, pvotes, representative)
					# Cluster update phase: replacing a prototype
					(proto_idx, prototypes, proto_dist, pvotes, representative, _buffer, repl) = prototypevoting_update(prototypes,seq,minimum_idx,proto_idx,seqcount,proto_dist,pvotes,representative,None)

					# Plot new sequence
					if RT_ANIMATION:
						ann_new = [x for x, y in enumerate(data)]	
						plot_seq(fig, 'SECLEDS', assigned_clusters, proto_idx, data, ann_new, pal,classes, pvotes)
						plt.show(block=False)
						plt.pause(1.0)
						plt.close()
					
					checkcount += len(seq)
					if len(buffer[pair][_len:]) > 0:
						buffer[pair] = buffer[pair][_len:]
					else:
						del buffer[pair]
					
					
				timer_current = time.time()
				if run_clus and (timer_current - timer_start) > save_every:
					print('\n--- Saving medoids', savecounter)
					print(proto_idx)
					
					with open(outfname,'a') as outfile:
						for cid,cluster in enumerate(prototypes):
							for pid,prototype in enumerate(cluster):
								row = [savecounter, cid, proto_idx[cid][pid], prototype]
								writer = csv.writer(outfile)
								writer.writerow(row)
					savecounter += 1
					print('--- End')
					plot_medoids('SECLEDS', timer_current, prototypes, nclasses, nprototypes, 0.0, DATASET, now_str)
					timer_start = time.time()
		x = sum([len(v) for (k,v) in buffer.items() if len(v) < _len])
		assert (overallcount == checkcount + x)
		
		print('\nFinal assigned clusters to stream', assigned_clusters)
		#print('-----')
		#for i,p in enumerate(prototypes):
		#	print('Cluster ', i)
		#	for _p in p:
		#		print(len(_p), _p)
		#print('-----')
		plot_data(data, assigned_clusters, classdict, pal, DATASET, now_str)
		



parser = argparse.ArgumentParser(description='SECLEDS: Real-time sequence clustering via k-medoids.')
parser.add_argument('k', type=int,  help='Number of clusters')
parser.add_argument('p', type=int,  help='Number of medoids/prototypes')
parser.add_argument('streamname', type=str, help='Name of the stream')
parser.add_argument('-ini', type=str, default='config.ini', help='Path/to/configuration/file')
parser.add_argument('streamFile', type=str, help='Path/to/file/containing/stream')
args = parser.parse_args()

nclasses = args.k
nprototypes = args.p
DATASET = args.streamname
inifile = args.ini
SAVED_PATH = args.streamFile

config = configparser.ConfigParser()
config.sections()
config.read(inifile)

RT_ANIMATION = config['EXP'].getboolean('realtime_animation')
batch_factor = float(config['EXP']['batch_factor'])
batchsize = int(nclasses*nprototypes*batch_factor)
save_every = int(config['REAL']['save_every'])
seq_len = int(config['REAL']['seq_length'])

GLOBALS.init()
now = datetime.now()
now_str = now.strftime("%d%m%y-%H%M%S")+'-plots'
os.mkdir(now_str)
outfname = now_str+'/medoids.csv'
outfile = open(outfname, 'w')
outfile.write('save_number, cluster, seqid, medoid\n')
outfile.close()

pal = sns.color_palette("hls", nclasses)


# Run stream
asyncio.run(streamfunc(SAVED_PATH, nclasses, nprototypes, batchsize, seq_len))


	