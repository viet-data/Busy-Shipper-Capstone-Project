import random as rd
import math 
import pylab as pl
import scipy.stats as st
import numpy as np
import folium
import heapq as hq
from IPython.display import display
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

#return the shortest path
def solution(parents, des):
    ans = []
    while des in parents:
        ans.append(des)
        des = parents[des]
    ans.append(des)
    return ans[::-1]

#A* algorithm
def A_star(G, ori, des):
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

class MST:
    def __init__(self, val, idx, nec, mat_dist, start):
        self.val = val
        self.idx = idx
        self.nec = nec
        self.mat_dist = mat_dist 
        self.span2 = {}
        self.edges = {}
        self.max = 0
        for i in nec:
            self.span2[i] = []
        self.start = start
        self.max_ = float('-inf')
        self.initialize()
    
    #depth first search
    def dfs(self, root, check, path):
        check.add(root)
        if root in path:
            c = 0
            for i in path[root]:
                if i != None and i in self.nec:
                    c += self.mat_dist[self.idx[self.val[root]]][self.idx[self.val[i]]]
                    self.max_ = max(self.max_, self.mat_dist[self.idx[self.val[root]]][self.idx[self.val[i]]])
                if i not in check:
                    c += self.dfs(i, check, path)
            return c
        else:
            return 0
    
    #return path cost
    def cost(self, path):
        check = set()
        c = 0
        for i in path:
            if i not in check:
                c += self.dfs(i, check, path)
        return c
    
    #Initialize minimum spanning tree
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
            self.max_ = max(self.max_, cost)
            if dep != None:
                self.edges[tuple(sorted([arr, dep]))] = cost
                span[dep].append(arr)
                self.span2[arr].append(dep)
                self.span2[dep].append(arr)
            used.add(arr)
            nec.discard(arr)
            del(nears[arr])
            for i in nec:
                if nears[i][0] > self.mat_dist[self.idx[self.val[arr]]][self.idx[self.val[i]]]:
                    nears[i] = (self.mat_dist[self.idx[self.val[arr]]][self.idx[self.val[i]]], arr)
                    nears[i] = (self.mat_dist[self.idx[self.val[i]]][self.idx[self.val[arr]]], arr)
    
    #Delete a node on minimum spanning tree
    def recall(self, curr, inplace=False):
        c = self.max
        for i in self.span2[curr]:
            c -= self.edges[tuple(sorted([curr, i]))]
        c = c + (len(self.span2[curr])-1)*self.max_
        if inplace:
            self.max = c
        return c 
    
    #Insert a node on minimum spanning tree
    def insert(self, curr):
        self.max -= (len(self.span2[curr])-1)*self.max_
        for i in self.span2[curr]:
            self.max += self.edges[tuple(sorted([curr, i]))]

#return path cost
def cost(mat_dist, path, idx):
    c = 0
    for i in range(0, len(path)-1):
        c += mat_dist[idx[path[i]]][idx[path[i+1]]]
    return c

#Check goal state
def isgoal(deliveried, cus):
    return len(deliveried) == len(cus)

#Generate next valid elements
def next_gent(nec, remain, purchased, val):
    nxt = []
    for i in nec:
        if i % 2 == 1 and remain > 0:
            nxt.append(i)
        if i % 2 == 0 and i+1 in purchased:
            nxt.append(i)
    return nxt

#iterative deepening-search
def DIS(i, nec, mst, d, purchased, deliveried, max_package, h, gamma, t):
    min_path = []
    nec_ = set()
    min_cost = [float('inf')]
    dis(i, nec, mst, d, [], min_path, nec_, 0, min_cost, purchased, deliveried, max_package, h, gamma, t)
    return min_path, nec_, min_cost[0]

#iterative deepening-search continue
def dis(curr, nec, mst, d, path, min_path, nec_, cost, min_cost, purchased, deliveried, max_package, h, gamma, t):
    nec.discard(curr)
    state = tuple(sorted(list(nec)))
    if state not in h:
        h[state] = mst.max
    if len(state) == 1:
        h[state] = 0
    if d == 0 or len(nec) == 0:
        c = cost + h[state]
        if min_cost[0] > c and (rd.choices([0, 1], weights=[1, 99])[0] or min_path == []):
            min_cost[0] = c 
            min_path.clear()
            min_path.extend(path)
            nec_.clear()
            nec_.update(nec)
        nec.add(curr)
        return
    nxt = next_gent(nec, max_package - (len(purchased) - len(deliveried)), purchased, mst.val)
    for i in nxt:
        c = cost + mst.mat_dist[mst.idx[mst.val[curr]]][mst.idx[mst.val[i]]]*gamma
        if c < min_cost[0]:
            path.append(i)
            mst.recall(i, inplace=True)
            if i % 2 == 0:
                deliveried.add(i)
            else:
                purchased.add(i)
            
            dis(i, nec, mst, d-1, path, min_path, nec_, c, min_cost, purchased, deliveried, max_package, h, gamma, t)
            mst.insert(i)
            if i % 2 == 0:
                deliveried.discard(i)
            else:
                purchased.discard(i)
            path.pop()
    nec.add(curr)

#Learning real time search
def LRTS(G, initial, sto, cus, mat_dist, lst, idx, max_package, gamma, T, d, limit):
    val = {2*i:cus[i] for i in range(len(cus))}
    val2 = {2*i+1:sto[i] for i in range(len(sto))}
    val[-2] = initial
    val.update(val2)

    purchased = set()
    deliveried = set()
    s = -2
    path = [-2]

    nec = set(range(2*len(sto)))
    nec.add(-2)
    ans = float('inf')
    opt_path = []
    count = 0
    curr = tuple(sorted(list(nec)))
    h = {curr:0, tuple():0}
    learned = 0
    t = [0]
    mst = MST(val, idx, nec, mat_dist, -2)
    while True:
        update = False
        failure = False
        while len(path) - 1 != len(sto) + len(cus):
            min_path, nec_, h_ = DIS(s, nec, mst, d, purchased, deliveried, max_package, h, gamma, t)
            l = 0
            if h[curr] < h_:
                update = True
                l = h_ - h[curr]
                h[curr] = h_
            if h_ == float('inf'):
                failure = True
            if learned + l <= T and not failure:    
                learned += l
                for pro in min_path:
                    mst.recall(pro, inplace=True)
                    if pro % 2 == 0:
                        deliveried.add(pro)
                    else:
                        purchased.add(pro)
                nec = nec_
                path.extend(min_path)
                curr = tuple(sorted(list(nec)))
                s = path[-1]
            else:
                if len(path) > 1:
                    pre = path.pop()
                    mst.insert(path[-1])
                    if pre % 2 == 0:
                        deliveried.discard(pre)
                    if pre % 2 == 1:
                        purchased.discard(pre)
                    nec.add(pre)
                    curr = tuple(sorted(list(nec | set([path[-1]]))))
                    s = path[-1]
                if len(path) == 1 and failure and h_ == float('inf'):
                    return path, float('inf')
                failure = False
        count += 1
        c = float('inf')
        if failure == False:
            path_ = []
            for i in path:
                path_.append(val[i])
            c = cost(mat_dist, path_, idx)
        if ans > c:
            ans = c
            opt_path = path_.copy()
        if count >limit :
            break
        if update :
            purchased = set()
            deliveried = set()
            s = -2
            path = [-2]
            nec = set(range(2*len(sto)))
            nec.add(-2)
            T = learned
            learned = 0
            mst = MST(val, idx, nec, mat_dist, 0)
    path = []
    for i in range(len(opt_path)-1):
        part = A_star(G, opt_path[i], opt_path[i+1])
        path.append(part)
    return path, ans

#Visualize optimal route
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