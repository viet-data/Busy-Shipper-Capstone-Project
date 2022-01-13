import heapq as hq
import osmnx as ox
import random as rd
import pylab as pl
import scipy.stats as st
import numpy as np
import folium
import heapq as hq
from IPython.display import display
import math


class MST:
    def __init__(self, val, idx, nec, mat_dist, start):
        self.val = val
        self.idx = idx
        self.nec = nec
        self.mat_dist = mat_dist 
        self.max = 0
        self.start = start
        self.initialize()

    def initialize(self):
        nears = {}
        for i in self.nec:
            nears[i] = (float('inf'), None)
        nears[self.start] = (0, None)
        used = set([None])
        nec = self.nec.copy()
        span = {}
        for i in self.nec:
            span[i] = []
        while len(used)-1 < len(self.nec):
            cost, arr, dep = float('inf'), None, None
            for i in nears:
                if (nears[i][1]  in used and cost > nears[i][0]) or arr == None:
                    cost = nears[i][0]
                    arr = i
                    dep = nears[i][1]
                    
            self.max += cost
            used.add(arr)
            nec.discard(arr)
            del(nears[arr])
            for i in nec:
                ax = min(self.mat_dist[self.idx[self.val[arr]]][self.idx[self.val[i]]], self.mat_dist[self.idx[self.val[i]]][self.idx[self.val[arr]]])
                if nears[i][0] > ax:
                    nears[i] = (ax, arr)

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

#return the shortest path
def solution(parents, des):
    ans = []
    while des in parents:
        ans.append(des)
        des = parents[des]
    ans.append(des)
    return ans[::-1]

#A* algorithm
def small_A_star(G, ori, des):
    frontier = [(0,ori)]
    checked = {ori:0}
    parents = {}
    while frontier:
        est, pos = hq.heappop(frontier)
        if pos == des:
            return solution(parents, des)
        for neighbor in G[pos]:
            cost = checked[pos] + G[pos][neighbor][0]['length']
            if neighbor not in checked or checked[neighbor] > cost:
                parents[neighbor] = pos
                checked[neighbor] = cost
                x1, y1 = G.nodes[pos]['x'], G.nodes[pos]['y']
                x2, y2 = G.nodes[neighbor]['x'], G.nodes[neighbor]['y']
                cost += haversine(y1, x1, y2, x2)
                hq.heappush(frontier, (cost, neighbor))
    return "Impossible"
                    
def path_of_cost(mat_dist, path, idx):
    c = 0
    for i in range(0, len(path)-1):
        c += mat_dist[idx[path[i]]][idx[path[i+1]]]
    return c

def next_gent(nec, state, max_package, val, val2):
    weigh = max_package + sum([1 if int(i) % 2 == 1 else -1 for i in nec])
    nxt = []
    for i in nec:
        if i in val2 and weigh > 0:
            nxt.append(i)
        if i not in val2 and i+1 not in nec:
            nxt.append(i)
    nxt = list(set(nxt))
    return nxt

def Astar(G, initial, sto, cus, mat_dist, idx, max_package):
    val = {2*i: cus[i] for i in range(len(cus))}
    val2 = {2*i+1: sto[i] for i in range(len(sto))}
    val[-2] = initial
    val.update(val2)
    nec = set(list(range(2*len(sto))))
    frontier = [(0, str(-2) + '_' + '_'.join([str(i) for i in sorted(list(nec))]))]
    parent = {}
    heuristic = {frontier[0][1]: 0}
    opt_path = []
    while frontier:
        #  The smallest in heap
        c, state = hq.heappop(frontier)
        
        # Check result 
        if '_' not in state:
            path = [val[int(state)]]
            child = state
            while child in parent:
                child = parent[child][0]
                path.append(val[int(child[:child.index('_')])])
            opt_path = path[::-1]
            path = []
            for i in range(len(opt_path)-1):
                part = small_A_star(G, opt_path[i], opt_path[i+1])
                path.append(part)
            return path, path_of_cost(mat_dist, opt_path, idx) 
        
        # id: current position & cost
        if '-2' not in state:
            id = int(state[:state.index('_')])
        else:
            id = -2
        # nec: remain position
        nec = set([int(i) for i in state[state.index('_')+1:].split('_')])
        heu = '_'.join([str(i) for i in sorted(list(nec))])
        cost = c - heuristic['_'.join([str(i) for i in sorted(list(nec)+[id])])]
        if heu not in heuristic:
            mst = MST(val, idx, nec, mat_dist, list(nec)[0])
            heuristic[heu] = mst.max
            
        # nxt: next states
        nxt = next_gent(nec, state, max_package, val, val2)
        for i in nxt:
            nec_ = nec.copy()
            nec_.discard(i)
            if len(nec_) != 0:
                state_ = str(i) + '_' + '_'.join(sorted([str(i) for i in nec_]))
            else:
                state_ = str(i)
            cost1 = cost + mat_dist[idx[val[id]]][idx[val[i]]]
            if state_ not in parent or parent[state_][1] > cost1:
                parent[state_] = [state, cost1]
                c = cost1 + heuristic[heu]
                hq.heappush(frontier, (c, state_))

# Visualize optimal route

def folium_route(G, paths, address, zoom, cus, sto):
    m = folium.Map(location=address, zoom_start=zoom)
    color = hex(0)[2:].zfill(6)
    d = 16711680//(len(paths)+10)
    visited = set()
    for path in paths:
        loc = []
        for i in path:
            x, y = G.nodes[i]['x'], G.nodes[i]['y']
            loc.append((y,x))
            if i in cus and i not in visited:
                        folium.Marker([y,x], popup='Customer', icon=folium.Icon(color="red")).add_to(m)
                        visited.add(i)
            if i in sto and i not in visited:
                folium.Marker([y,x], popup='Store', icon=folium.Icon(color="green")).add_to(m)
                visited.add(i)
        folium.PolyLine(loc, color="#"+color, weight=8).add_to(m)
        color = hex(int(color, 16) + d)[2:].zfill(6)
        
    start = paths[0][0]
    x, y = G.nodes[start]['x'], G.nodes[start]['y']
    folium.Marker([y,x], popup='Start', icon=folium.Icon(color="black")).add_to(m)
    end = paths[-1][-1]
    x, y = G.nodes[end]['x'], G.nodes[end]['y']
    folium.Marker([y,x], popup='Finish', icon=folium.Icon(color="blue")).add_to(m)
    display(m)