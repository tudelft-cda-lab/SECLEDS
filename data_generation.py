import numpy as np
import random
from sklearn.datasets import make_blobs, make_circles
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from dtw import dtw
from collections import Counter
import sys, os, re, glob

# Variant 1: sine curve dataset
def generateCurve(n, _freq, err, phase):
    trajectory = []
    params = []
    n = int(n)
    for i in range(n):
        line = np.arange(1, 101, 1)
        freq = _freq[0]
        if len(_freq) > 1:
            freq = random.uniform(_freq[0], _freq[1])
        error = [random.random() * err for x in range(len(line))]
        l = np.sin((freq * line) + phase) + error
        trajectory.append(l)
        params.append((freq, err, phase))
    # print(len(data))
    return (trajectory, params)  # random sine curve + random error

# Variant 2: handwritten dataset
def parseCharacterFile(lines):
    points = dict()
    newchar = False
    cont = False
    point = []
    cclass = None

    # classes_ = [x.strip('"') for x in lines[0].split(' ')][1:-1]
    for line in lines[1:]:
        if '.COMMENT' in line and 'Class' in line and '[' in line and '#' not in line:
            b = re.findall('.*?\.COMMENT\s+Class\s+\[(.*?)\]', line)
            cclass = b[0]
            # print(cclass)
            newchar = True
            point = []
            continue
        if '.PEN_UP' in line:
            cont = False
            if cclass not in points.keys():
                points[cclass] = []
            points[cclass].append(point)
        if '.PEN_DOWN' in line:
            cont = True
            continue
        if newchar and cont:
            b = re.findall('.*?(\d+)\s+([-\d]+).*', line)
            # print(line)
            xy = b[0]
            point.append((int(xy[0]), int(xy[1])))
    return points

# Variant 3: network traffic
def parse_netflows(path, thresh=100):
    previousTimestamp = {}
    connections = {}
    labels = {}
    print('reading... ', os.path.basename(path))
    f = open(path, 'r')
    netflows = f.readlines()
    first = datetime.strptime((netflows[1].split(','))[0], '%Y/%m/%d %H:%M:%S.%f')
    last = datetime.strptime((netflows[len(netflows) - 1].split(','))[0], '%Y/%m/%d %H:%M:%S.%f')
    tot_duration = last - first
    print('TOTAL DURATION ', tot_duration)

    for nid, netflow in enumerate(netflows):  # [:int(len(netflows)/2)]):
        if nid == 0:
            continue

        flow = netflow.split(',')
        try:
            startTime = flow[0]
            duration = float(flow[1])
            proto = flow[2]
            src_ip = flow[3]
            sport = int(flow[4], 0) if flow[4] != '' else -1
            direction = flow[5]
            dst_ip = flow[6]
            dport = int(flow[7], 0) if flow[7] != '' else -1
            totpkts = int(flow[11])
            totbytes = int(flow[12])
            avg_bytes = round(totbytes / float(totpkts), 3)
            label = (flow[14])[5:-1]
            # Opt1: Mark netflows as malicious
            if 'botnet' in label.lower():
                label = 'botnet'
            elif 'normal' in label.lower():
                label = 'normal'
            elif 'background' in label.lower():
                label = 'background'
                continue
            else:
                label = 'unknown'
                continue
        except:
            print('issue parsing', flow)
            continue

        timestamp = datetime.strptime(startTime, '%Y/%m/%d %H:%M:%S.%f')

        key = src_ip
        iat = 0
        if key in previousTimestamp:
            iat = (timestamp - previousTimestamp[key]).microseconds / 1000.0
        else:
            iat = 0
        previousTimestamp[key] = timestamp
        tupple = avg_bytes  # options: duration#totbytes#avg_bytes#duration#, sport, dport)
        if key not in connections.keys():
            connections[key] = []
        connections[key].append(tupple)

        if key not in labels.keys():
            labels[key] = []
        labels[key].append(label)
    f.close()

    todel = []
    lengths = [len(x) for x in connections.values()]
    print('\t', os.path.basename(path), " # sequences: ", len(labels))
    print('\tBefore cleanup: ', len(connections), ' connections/sequences.')
    print('\tAvg connection length: ', round(sum(lengths) / len(lengths), 3), 'Shortest: ', min(lengths), 'Longest: ',
          max(lengths))
    for i, v in connections.items():  # clean it up
        if len(v) < thresh:
            todel.append(i)
    for item in todel:
        del connections[item]
        del labels[item]

    lengths = [len(x) for x in connections.values()]
    print('\tAfter cleanup: ', len(connections), ' connections/sequences.')
    print('\tAvg connection length: ', round(sum(lengths) / len(lengths), 3), 'Shortest: ', min(lengths), 'Longest: ',
          max(lengths))

    # In correct format (list of connections and label for each connection)
    data, true_labs, meta = [], [], []
    outfile = open(path + '-meta.txt', 'w')

    for key, lab in labels.items():
        conns = split_on_window(connections[key], thresh)
        labs = split_on_window(lab, thresh)

        for cid, (lab_, conn_) in enumerate(zip(labs, conns)):
            real_label = None
            if 'botnet' in lab_:
                real_label = 'Botnet'
            else:
                real_label = 'Normal'  # max(lab_,key=lab_.count) # whichever is the most frequent label

            data.append(list(conn_))
            true_labs.append(real_label)
            meta.append(str(cid) + '|' + real_label[0])
            outfile.write(key + '-' + str(cid) + ' , ' + real_label + '\n')
    outfile.close()
    print('\tAfter sliding windows: Total flows: ', len(data), ' connections/sequences.')
    print('\tClass distro ', Counter(true_labs).items())
    return (data, true_labs, meta)


# Read points (blobs, circles)
def read_points(n_samples, nclasses, saved_path):
    segments = []
    labs = []
    files = glob.glob(saved_path + '/*_*')  # Path to the saved dataset
    for f in files:
        lab = f.split('_')[-1]
        if int(lab) >= nclasses:
            continue
        cname = []
        f_ = open(f, 'r')
        lines = f_.readlines()[:int(n_samples / nclasses)]
        for line in lines:
            elements = line.split(';')
            elements = [float(x) for x in elements]
            cname.append(elements)
        segments.append(cname)
        labs.append([int(lab)] * len(cname))
        f_.close()

    classes = list(range(nclasses))
    classdict = {k: v for v, k in enumerate(classes)}

    trajectory = []
    print('# Points per class', [len(x) for x in segments])

    for i, cx in enumerate(segments):
        trajectory.extend(list(zip(cx, labs[i])))
    
    random.shuffle(trajectory)

    print('# Total points', len(trajectory))

    X = [x for (x, y) in trajectory]  # Data
    ann = [x for x, y in enumerate(X)]  # IDs
    labs = [y for (x, y) in trajectory]  # classes
    meta = [str(i) + '|' + str(l) for i, l in zip(ann, labs)]
    dist = X  # get_distmatrix(X, 'stat')

    return (X, ann, labs, dist, classdict, meta)

# Read univariate (sine curve)
def read_curves(n_samples, nclasses, saved_path):
    params, segments = {}, []
    files = glob.glob(saved_path + '/*_*')  # Path to the saved dataset
    for f in files:
        lab = f.split('_')[-1]
        if int(lab) >= nclasses:
            continue
        cname = []
        f_ = open(f, 'r')
        lines = f_.readlines()[:int(n_samples / nclasses)]
        for line in lines:
            elements = line.split(';')
            elements = [float(x) for x in elements]  # [:50]
            cname.append(elements)
        segments.append(cname)
        f_.close()
    try:
        files = glob.glob(saved_path + '/*params*')  # Path to the saved dataset
        for f in files:
            lab = f.split('params')[-1]
            if int(lab) >= nclasses:
                continue
            cname = []
            f_ = open(f, 'r')
            lines = f_.readlines()[:int(n_samples / nclasses)]
            for line in lines:
                elements = line.split(';')
                elements = tuple([float(x) for x in elements])  # [:50]
                cname.append(elements)
            params[int(lab)] = cname
            f_.close()
    except:
        print('Param info not available')

    classes = list(range(nclasses))
    classdict = {k: v for v, k in enumerate(classes)}

    # Preparing input sequence data (Exp with different settings, e.g. randomize, inverted, etc)
    trajectory = []

    print('# Sequences per class', [len(x) for x in segments])
    print('Sequence lengths average ', [sum([len(x) for x in seq]) / float(len(seq)) for seq in segments])

    if params == {}:
        for i, cx in enumerate(segments):
            trajectory.extend(list(zip(cx, [classes[i]] * len(cx))))
    else:
        for i, cx in enumerate(segments):
            trajectory.extend(list(zip(cx, [classes[i]] * len(cx), params[i])))

    random.shuffle(trajectory)

    # First kp points are the prototypes
    print('# Total sequences', len(trajectory))

    if params != {}:
        X = [x for (x, y, z) in trajectory]  # Data
        ann = [x for x, y in enumerate(X)]  # IDs
        labs = [y for (x, y, z) in trajectory]  # classes
        params = [z for (x, y, z) in trajectory]

    else:
        X = [x for (x, y) in trajectory]  # Data
        ann = [x for x, y in enumerate(X)]  # IDs
        labs = [y for (x, y) in trajectory]  # classes
    meta = [str(i) + '|' + str(l) for i, l in zip(ann, labs)]

    dist = X  # get_distmatrix(X)

    return (X, ann, labs, dist, classdict, params, meta)

# Read multivariate (handwritten)
def read_chars(classes, saved_path):
    LIM = 10  # limit how many samples per class
    classdict = {k: v for v, k in enumerate(classes)}
    segments = dict()
    files = glob.glob(saved_path + '/*')  # Path to dataset

    for f in files:
        f_ = open(f, 'r')
        lines = f_.readlines()
        content = parseCharacterFile(lines)
        for cclass, segment in content.items():
            if cclass not in classes:
                continue
            if cclass not in segments.keys():
                segments[cclass] = []
            # if len(segments[cclass]) > LIM:
            #    continue
            segments[cclass].extend(segment)
        f_.close()

    trajectory = []

    print('# Sequences per class', [len(x) for x in segments.values()])
    print('Sequence lengths average ', [sum([len(x) for x in seq]) / float(len(seq)) for seq in segments.values()])

    for i, cx in enumerate(segments.values()):
        trajectory.extend(list(zip(cx, [classes[i]] * len(cx))))

    random.shuffle(trajectory)

    print('# Total sequences', len(trajectory))

    X = [x for (x, y) in trajectory]  # Data
    ann = [x for x, y in enumerate(X)]  # IDs
    labs = [y for (x, y) in trajectory]  # classes
    meta = [str(i) + '|' + l for i, l in zip(ann, labs)]
    # for trajectory data, we need to compute pairwise distances to view, otherwise it doesnt work
    print('Overall min:', min([len(x) for x in X]))
    print('Overall max:', max([len(x) for x in X]))

    dist = get_distmatrix(X)

    return (X, ann, labs, dist, classdict, meta)

# Read multivariate (network traffic)
def read_traffic(nclasses, path):
    netflows, labels, metadata = [], [], []
    files = glob.glob(path + '/*.binetflow')  # Path to dataset
    print('About to read netflows...')
    for f in files:
        netflows_, labels_, meta_ = parse_netflows(f)
        netflows.extend(netflows_)
        labels.extend(labels_)
        metadata.extend(meta_)
    print('Done reading netflows.')

    numbers = [x for x in range(nclasses)]
    if len(labels) < nclasses:
        labels.extend(numbers[len(labels):])
    classdict = {k: v for v, k in enumerate(set(labels))}

    print('Class distro ', Counter(labels).items())

    X = [x for x in netflows]  # Data
    ann = [x for x, y in enumerate(X)]  # IDs
    labs = [x for x in labels]  # classes

    dist = X  # get_distmatrix(data, 'multi')

    print('Total sequences ', len(netflows), ' with fixed length ', len(netflows[0]))

    return (X, ann, labs, dist, classdict, metadata)


# Create data: points
def make_blobs(n_samples, nclasses, now_str):
    segments = []
    labs = []
    std = [random.uniform(0, 1) for x in range(nclasses)]
    X, y_true = make_blobs(n_samples=n_samples, centers=nclasses, cluster_std=std, random_state=42)
    for cclass in range(nclasses):
        c1 = []
        y1 = []
        for (x, y) in zip(X, y_true):
            if cclass == y:
                c1.append(x)
                y1.append(y)
        segments.append(c1)
        labs.append(y1)
    # write this dataset in a file
    path = 'datasets/blobs/' + now_str[:-6]
    if not os.path.exists(path):
        os.makedirs(path)
    for cid, segment in enumerate(segments):
        fi = open(path + '/blobs_' + str(cid), 'w')
        for seg in segment:
            fi.write(';'.join([str(x) for x in seg]))
            fi.write('\n')
        fi.close()
    print('Blobs generated.')
    return

def make_circles(n_samples, nclasses, now_str):
    if nclasses > 3:
        print('More than 3 classes not supported at this time')
        sys.exit()
    segments = []
    labs = []
    X, y_true = [], []
    numcalls = nclasses - 1  # math.ceil(nclasses/2)
    for i in range(numcalls):
        noise = random.uniform(0, 0.1)
        factors = np.arange(0.1, 0.9, 0.1)
        factor = random.choice(factors)

        circles, Y_circles = make_circles(n_samples=(int(n_samples / (2 * nclasses)), int(n_samples / nclasses)),
                                          random_state=3, noise=noise, factor=factor)
        if i > 0:
            Y_circles[Y_circles == 1] = i + 1
        X.extend(circles)
        y_true.extend(Y_circles)
    for cclass in range(nclasses):
        c1 = []
        y1 = []
        for (x, y) in zip(X, y_true):
            if cclass == y:
                c1.append(x)
                y1.append(y)
        segments.append(c1)
        labs.append(y1)
    # write this dataset in a file
    path = 'datasets/circles/' + now_str[:-6]
    if not os.path.exists(path):
        os.makedirs(path)
    for cid, segment in enumerate(segments):
        fi = open(path + '/circles_' + str(cid), 'w')
        for seg in segment:
            fi.write(';'.join([str(x) for x in seg]))
            fi.write('\n')
        fi.close()
    print('Circles generated.')
    return

# Create data: sine curve
def make_curves(n_samples, nclasses, now_str):
    samplingrate = 1
    segments = []
    freqs = [(0.1, 0.12), (0.2, 0.22), (0.42, 0.44), (0.6, 0.66)]
    errs = [0.2, 0.4, 0.7, 0.1]
    phases = [5, 12, -10, -20]
    # freqs, errs, phases = set(), set(), set()
    params = {}
    meta = []

    for i in range(nclasses):
        freq, err, phase = None, None, None
        '''while True:  
            f = random.uniform(0, 1)          
            freq = (f, f+0.02)
            print('try freq', freq)
            if not _in_(freq, freqs):
                freqs.add(freq)
                break
        while True:
            err = random.uniform(0, 1)
            print('try err', err)
            if not _in_(err, errs):
                errs.add(err)
                break
        while True:
            phase = int(random.uniform(-15, 15))
            print('try', phase)
            if not _in_(phase, phases):
                phases.add(phase)
                break     
        print('Params, ', freq, err, phase)'''
        freq = freqs[i]
        err = errs[i]
        phase = phases[i]
        c1, p = generateCurve(n_samples / nclasses, freq, err, phase)
        segments.append(c1)
        params[i] = p
    # write this dataset in a file
    path = 'datasets/sine-curve/' + now_str[:-6]
    if not os.path.exists(path):
        os.makedirs(path)
    for cid, segment in enumerate(segments):
        fi = open(path + '/sine-curve_' + str(cid), 'w')
        for seg in segment:
            fi.write(';'.join([str(x) for x in seg]))
            fi.write('\n')
        fi.close()
    # Store params in file too
    path = 'datasets/sine-curve/' + now_str[:-6]
    if not os.path.exists(path):
        os.makedirs(path)
    for cid, params_ in params.items():
        fi = open(path + '/sine-curve-params' + str(cid), 'w')
        for param in params_:
            fi.write(';'.join([str(x) for x in param]))
            fi.write('\n')
        fi.close()
    print('Curves generated')
    return


# Helper functions
def _in_(new, past):
    flag = False
    thresh = 0.2
    if isinstance(new, tuple):
        for p in past:
            if abs(new[0] - p[0]) < thresh and abs(new[1] - p[1]) < thresh:
                flag = True
                break
    else:
        if new > 1:
            thresh = 2
        else:
            thresh = 0.1
        for p in past:
            if abs(new - p) < thresh:
                flag = True
                break

    return flag

def split_on_window(sequence, limit):
    iterators = [iter(sequence[index:][::int(limit / 2)]) for index in range(limit)]
    return zip(*iterators)

def get_distmatrix(data, data_type="seq"):
    print('Computing pairwise distance matrix...')
    dist = [-1] * len(data)
    dist = [[-1] * len(data) for i in dist]

    for i in range(len(data)):
        for j in range(i + 1):
            _d = None
            if i == j:
                dist[i][j] = 0.0
            elif i > j:
                _d = 0
                if data_type == "seq":
                    _d = dtw(data[i], data[j], distance_only=True, dist_method="euclidean").distance
                elif data_type == "multi":
                    _d, d = 0, 0
                    num_feat = len(data[i])
                    for fid in range(num_feat):
                        a = [x[fid] for x in data[i]]
                        b = [x[fid] for x in data[j]]
                        if fid in [0, 1]:
                            d = euclidean_distances(a, b)[0][0]
                        else:
                            servs = set(a)
                            servs.update(set(b))
                            a_ = [(1 if x in a else 0) for x in servs]
                            b_ = [(1 if x in b else 0) for x in servs]
                            d = cosine_similarity(a, b)
                        _d += d
                    _d = _d / num_feat
                else:
                    _d = euclidean_distances(np.array(data[i]).reshape(1, -1), np.array(data[j]).reshape(1, -1))[0][0]
                dist[i][j] = _d
                dist[j][i] = _d
    dist = np.array([np.array(xi) for xi in dist])
    return dist
