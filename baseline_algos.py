KMedoids = None
CluStream = None
STREAMKMeans = None
MiniBatchKMeans = None

# banditPAM baseline
def Bpam_cluster(exp_data, nclasses, metadata):
    global KMedoids
    if not KMedoids:
        from banditpam import KMedoids
    bandit_kmed = KMedoids(n_medoids=nclasses, algorithm="BanditPAM")
    z = bandit_kmed.fit(exp_data, 'L2', 'mnist')
    bandit_labels = [int(x) for x in bandit_kmed.labels]
    prototypes = [[exp_data[int(p)]] for p in bandit_kmed.medoids]
    proto_idx = [[int(p)] for p in bandit_kmed.medoids]
    points = bandit_kmed.medoids
    meta = None
    if metadata != {}:
        meta = [[metadata[int(p)]] for p in bandit_kmed.medoids]

    return (points, bandit_labels, prototypes, proto_idx, meta)

# CluStream baseline
def CluStream_init(data, labs, nprototypes, nclasses, classdict):
    global CluStream
    if not CluStream:
        from river.cluster import CluStream
    prototypes = []
    proto_idx = []
    representative = None
    assigned_clusters = []
    CluStream_ = CluStream(time_window=1, \
                                  max_micro_clusters= 5*nprototypes*nclasses, \
                                  n_macro_clusters=nclasses, \
                                  halflife=0.5)
    for d in data:
        d = dict(zip(range(len(d)), d))
        CluStream_ = CluStream_.learn_one(d)
        pred = int(CluStream_.predict_one(d))
        assigned_clusters.append(pred)
    pvotes = [[0 ] *nprototypes for x in range(nclasses)]
    return (prototypes, proto_idx, assigned_clusters, pvotes, CluStream_)

def CluStream_cluster(prototypes, point, assigned_clusters, pvotes, CluStream_):
    d = dict(zip(range(len(point)), point))
    CluStream_ = CluStream_.learn_one(d)
    pred = int(CluStream_.predict_one(d))
    assigned_clusters.append(pred)
    return (assigned_clusters, pvotes, CluStream_)

# StreamKM baseline
def StreamKM_init(data, labs, nprototypes, nclasses, classdict):
    global STREAMKMeans
    if not STREAMKMeans:
        from river.cluster import STREAMKMeans
    prototypes = []
    proto_idx = []
    representative = None
    assigned_clusters = []
    KMEANS = STREAMKMeans(chunk_size=1, \
                                  n_clusters=nclasses, \
                                  halflife=0.5)
    for d in data:
        d = dict(zip(range(len(d)), d))
        KMEANS = KMEANS.learn_one(d)
        pred = int(KMEANS.predict_one(d))
        assigned_clusters.append(pred)
    pvotes = [[0 ] *nprototypes for x in range(nclasses)]
    return (prototypes, proto_idx, assigned_clusters, pvotes, KMEANS)

def StreamKM_cluster(prototypes, point, assigned_clusters, pvotes, KMEANS):
    d = dict(zip(range(len(point)), point))
    KMEANS = KMEANS.learn_one(d)
    pred = int(KMEANS.predict_one(d))
    assigned_clusters.append(pred)
    return (assigned_clusters, pvotes, KMEANS)

# MiniBatch baseline
def MiniBatch_init(data, labs, nprototypes, nclasses, classdict):
    global MiniBatchKMeans
    if not MiniBatchKMeans:
        from sklearn.cluster import MiniBatchKMeans
    prototypes = []
    proto_idx = []
    representative = None
    assigned_clusters = []
    KMEANS = MiniBatchKMeans(init='k-means++', \
                             n_clusters=nclasses, \
                             batch_size=1, \
                             max_iter=1)
    KMEANS.fit(data)
    pred = KMEANS.predict(data)
    pred = [int(x) for x in pred]
    assigned_clusters.extend(pred)
    pvotes = [[0 ] *nprototypes for x in range(nclasses)]

    return (prototypes, proto_idx, assigned_clusters, pvotes, KMEANS)

def MiniBatch_cluster(prototypes, point, assigned_clusters, pvotes, KMEANS):
    KMEANS = KMEANS.partial_fit([point])
    pred = int(KMEANS.predict([point])[0])
    assigned_clusters.append(pred)
    return (assigned_clusters, pvotes, KMEANS)
