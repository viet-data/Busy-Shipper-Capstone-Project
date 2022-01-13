import numpy as np
import matplotlib.pyplot as plt
import heapq as hq
import math

#fomulation to calculate distance on Sphere
def haversine(lat1, lon1, lat2, lon2):
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2));
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c

#Sort a point in map in ascending order of standard deviation
def arrange(cus, sto, G, graph=False):
    lst = list(set(cus + sto))
    mat = [[0]*len(lst) for i in range(len(lst))]
    for i in range(len(lst)):
        for j in range(len(lst)):
            x1, y1 = G.nodes[lst[i]]['x'], G.nodes[lst[i]]['y']
            x2, y2 = G.nodes[lst[j]]['x'], G.nodes[lst[j]]['y']
            mat[i][j] = haversine(y1, x1, y2, x2)
    m_dist = mat
    mat = np.array(mat, dtype='float')
    mat = mat.std(axis=1).transpose()
    std = [(mat[i], i) for i in range(len(lst))]
    std.sort(reverse=True)
    std = [i[1] for i in std]
    nxt = []
    for i in range(len(lst)):
        arr = [(m_dist[i][j], j) for j in range(len(lst))]
        arr.sort(reverse=True)
        nxt.append([j[1] for j in arr])
    
    if graph:
        plt.bar(np.linspace(0, len(lst)-1, len(lst)), mat)
        plt.show()
    return lst, std, nxt

#return path cost
def path_cost(path, G):
        cost = 0
        for i in range(len(path)-1):
            cost += G[path[i]][path[i+1]][0]['length']
        return cost

#Uniform cost search algorithm
def uniform_cost(G, initial, goal, known, lst,idx, mat_dist):
    frontier = [(0, initial)]
    parents = {}
    checked = {initial:0}
    while frontier:
        c, node= hq.heappop(frontier)
        if node in idx and (idx[initial], idx[node]) not in known:
            mat_dist[idx[initial]][idx[node]] = c
            known.add((idx[initial], idx[node]))
        if node == goal:
            return 
        for neighbor in G[node]:
            cost = c + G[node][neighbor][0]['length']
            if neighbor not in checked or checked[neighbor] > cost:
                checked[neighbor] = cost
                parents[neighbor] = node
                hq.heappush(frontier, (cost, neighbor))

#return distance matrix between locations that are store, customer, or initial position
def getdist(lst, std, nxt, G):
    mat_dist = [[float('inf')]*len(lst) for i in range(len(lst))]
    idx = {lst[i]: i for i in range(len(lst))}
    known = set()
    for i in std:
        for j in nxt[i]:
            if i != j and (i, j) not in known:
                uniform_cost(G, lst[i], lst[j], known, lst, idx, mat_dist)
            else:
                if i == j:
                    mat_dist[i][j] = 0
    return mat_dist