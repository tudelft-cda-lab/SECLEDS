def init():

    global APPROX, PLOT, DEBUG, count_dist, BUFFER_LIM, NEWEST, FARTHEST, CLOSEST
    
    count_dist = 0
    
    APPROX = False
    PLOT= False
    DEBUG = False
    BUFFER_LIM = 2
    NEWEST = {} # clusterID, medID
    FARTHEST = {} # to the newest point in the cluster
    CLOSEST = {}
    
