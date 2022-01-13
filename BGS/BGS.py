import math
import random as rd
import heapq as hq
import time
import pylab as pl
import scipy.stats as st
import numpy as np
from IPython.display import display
import folium
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

class GA:
    def __init__(self, n, val, gen, num_pop, nec, purchased, deliveried, s, max_package):
        self.purchased = purchased
        self.deliveried = deliveried
        self.nec = nec
        self.val = val
        self.max_package = max_package
        self.block = int(math.log(n)/math.log(2)) + 1
        self.values = {}
        self.keys = {}
        for i in range(n):
            str_ = bin(i)[2:].zfill(self.block)
            self.keys[i] = str_
            self.values[str_] = i
        self.pop = self.population(list(nec), num_pop, s, 0)
        self.gen = gen
        self.num_pop = num_pop
    
    #return cost of a feasible solution
    def cost(self, mat_dist, path, idx):
        lst = []
        i = 0
        while i < len(path):
            key = self.values[path[i:i+self.block]]
            lst.append(self.val[key])
            i += self.block
        c = 0
        for i in range(0, len(lst)-1):
            c += mat_dist[idx[lst[i]]][idx[lst[i+1]]]
        return c
    
    #check the validity of a solution
    def check(self, path, curr):
        count = curr
        used = set()
        for i in path: 
            if i % 2 == 1:
                if i - 1 in used:
                    return False
                count += 1
            else:
                used.add(i)
                count -= 1
            if count > self.max_package:
                return False
        return True 
    
    #return a random population
    def population(self, remain, num_pop, start, curr):
        used = set()
        initial = []
        count = curr
        limit = 10000
        if self.max_package == 1:
            lst = remain.copy()
            pop = []
            while num_pop > 0:
                resident = [start]
                rd.shuffle(lst)
                for i in range(len(lst)):
                    if lst[i] % 2 == 0:
                        if lst[i] + 1 in resident:
                            idx = resident.index(lst[i]+1)
                            if idx < len(resident)-1:
                                resident.insert(idx+1, lst[i])
                            else:
                                resident.append(lst[i])
                        else:
                            resident.append(lst[i])
                    else:
                        if lst[i] - 1 in resident:
                            idx = resident.index(lst[i]-1)
                            resident.insert(idx, lst[i])
                        else:
                            resident.append(lst[i])
                str_ = ''
                for j in resident:
                    str_ += self.keys[j]
                pop.append(str_)
                num_pop -= 1
            return pop
        sto = sorted([i for i in remain if i % 2 == 1])
        cus = sorted([i for i in remain if i % 2 == 0])
        initial = []
        i = 0
        j = 0
        if curr >= self.max_package:
            initial.append(cus[j])
            j += 1
            count -= 1
        while i < len(sto):
            if count < self.max_package:
                initial.append(sto[i])
                used.add(sto[i])
                count += 1
                i += 1
            if j < len(cus):
                initial.append(cus[j])
                count -= 1
                j += 1
        while j < len(cus):
            initial.append(cus[j])
            j += 1
        i = 0
        pop = []
        if self.check(initial, curr) == False:
            return False
        condition = (curr >= self.max_package)
        k = 0
        while i < num_pop:
            a = rd.randint(0, len(initial)-1)
            b = rd.randint(0, len(initial)-1)
            new = initial.copy()
            new[a], new[b] = new[b], new[a]
            if condition and new[0] % 2 == 1:
                continue
            if self.check(new, curr):
                str_ = self.keys[start]
                for j in new:
                    str_ += self.keys[j]
                pop.append(str_)
                i += 1
                initial = new
            k += 1
            if k > limit:
                if i == num_pop:
                    return pop
                else:
                    str_ = self.keys[start]
                    for j in p:
                        str_ += self.keys[j]
                    return pop + [str_]*(num_pop - len(pop))
                break
        return pop
    
    #Get parents for mating
    def parents(self, pop, mat_dist, idx):
        costs = [(self.cost(mat_dist, x, idx), x) for x in self.pop]
        costs.sort()
        self.pop = [x[1] for x in costs]
        return pop[:self.num_pop//2]
    
    #Combine parents to get children and add new members if the number of population is less than the dafault value num_pop
    def mate(self, parents):
        i = 0
        children = []
        half = len(self.nec)//2
        d = 2
        while i + d < len(parents) :
            x = []
            y = []
            for j in range(0, len(parents[i]), self.block):
                x.append(self.values[parents[i][j:j+self.block]])
                y.append(self.values[parents[i+1][j:j+self.block]])
            child1 = x[:half]
            used = set(child1)
            child2 = []
            for j in y:
                if j not in used:
                    if j % 2 == 1 and j - 1 in used:
                        child2.append(j)
                    else:
                        child1.append(j)
                        used.add(j)
                else:
                    child2.append(j)
            child2 += x[half:]
            if self.check(child1[1:], len(self.purchased) - len(self.deliveried)):
                str_ = ''
                for j in child1:
                    str_ += self.keys[j]
                children.append(str_)
            if self.check(child2[1:], len(self.purchased) - len(self.deliveried)):
                str_ = ''
                for j in child2:
                    str_ += self.keys[j]
                children.append(str_)
            i += d
        nec_ = self.nec.copy()
        curr = self.values[parents[0][:self.block]]
        pop = self.population(list(nec_), self.num_pop - len(parents) - len(children), curr, len(self.purchased) - len(self.deliveried))
        for i in pop:
            new = []
            for j in range(0, len(i), self.block):
                new.append(self.values[i[j:j+self.block]])
            if self.check(new[1:], len(self.purchased) - len(self.deliveried)) == False:
                print('POP DOES Not True')
        return parents + children + pop
    
    #mutate some elements in population 
    def mutate(self, pop, rate=0.01):
        if len(self.nec) <= 1:
            return
        n = int(len(pop)*rate)
        for i in range(n):
            pos = rd.randint(0, len(pop)-1)
            pos_m = rd.randint(0, len(self.nec))+1
            change = rd.randint(0, len(self.nec))+1
            #print(len(self.nec), pos_m, change, len(pop[pos]))
            if pos_m == change:
                continue
            if pos_m > change:
                pos_m, change = change, pos_m
            first = pop[pos][pos_m*self.block: (pos_m+1)*self.block]
            second = pop[pos][change*self.block: (change+1)*self.block]
            
            part1 = pop[pos][:pos_m*self.block]
            part2 = pop[pos][(pos_m+1)*self.block:change*self.block]
            part3 = pop[pos][(change+1)*self.block:]
            new_gen = part1 + second + part2 + first + part3
            new = []
            
            for j in range(0, len(new_gen), self.block):
                new.append(self.values[new_gen[j:j+self.block]])
            if self.check(new[1:], len(self.purchased) - len(self.deliveried)):
                pop[pos] = new_gen
                
    #filter population to choose elements that the second node is chose 
    #and add new members to the number of population is num_pop
    def filter(self, chose):
        new = []
        for i in self.pop:
            first = self.values[i[self.block:self.block*2]]
            if first == chose:
                new.append(i[self.block:])
        pop = self.population(list(self.nec), self.num_pop - len(new), chose, len(self.purchased) - len(self.deliveried))
        if pop == False:
            return False
        max_ = max([len(p) for p in new])
        self.pop = new + pop

    # Apply Genetic algorithm, choose the best path, get the second node of this solution and return it for the next move in BGS search 
    def GAP(self, mat_dist, idx, path, min_path, min_cost, curr_cost):
        if self.pop == False:
            return 'Failure'
        gen = self.gen
        best = []
        min_ = float('inf')
        while gen >= 0:
            parents = self.parents(self.pop, mat_dist, idx)
            self.pop = self.mate(parents)
            if self.pop == False:
                if min_ + curr_cost < min_cost:
                    curr_best = []
                    i = best
                    for j in range(0, len(i), self.block):
                        curr_best.append(self.values[i[j:j+self.block]])
                    min_path.clear()
                    min_path.extend(path[:-1] + curr_best)
                return "Failure"
            self.mutate(self.pop, rate=0.01)
            gen -= 1
        costs = [(self.cost(mat_dist, x, idx), x) for x in self.pop]
        costs.sort()
        self.pop = [x[1] for x in costs]
        c = self.cost(mat_dist, self.pop[0], idx)
        if c < min_:
            min_ = c
            best = self.pop[0]
        chose = None
        new = []
        for i in self.pop:
            first = self.values[i[self.block:self.block*2]]
            if first % 2 == 0:
                if first not in self.deliveried and first+1 in self.purchased:
                    self.deliveried.add(first)
                    self.nec.discard(first)
                    chose = first
                    break
            else:
                if first not in self.purchased and len(self.purchased) - len(self.deliveried) < self.max_package:
                    chose = first
                    self.purchased.add(first)
                    self.nec.discard(first)
                    break
        
        if chose == None:
            return "Failure"
        else:
            if min_ + curr_cost < min_cost[0]:
                curr_best = []
                i = best
                for j in range(0, len(i), self.block):
                    curr_best.append(self.values[i[j:j+self.block]])
                min_path.clear()
                min_path.extend(path[:-1] + curr_best)
                min_cost[0] = min_ + curr_cost
            if self.filter(chose) == False:
                return "Failure"
            
            return chose

        import time

#return cost of a solution
def cost(mat_dist, path, idx, val):
    ans = []
    for i in path:
        ans.append(val[i])
    path = ans
    c = 0
    for i in range(0, len(path)-1):
        c += mat_dist[idx[path[i]]][idx[path[i+1]]]
    return c

#Best genetic algorithm
def BGS(cus, sto, s, m, G, mat_dist, idx, limit_num, limit_time, num_gen, num_pop):
    val = {2*i:cus[i] for i in range(len(cus))}
    val2 = {2*i+1:sto[i] for i in range(len(sto))}
    val[len(cus)+len(sto)] = s
    val.update(val2)

    purchased = set()
    deliveried = set()
    start = len(cus)+len(sto)
    path = [len(cus)+len(sto)]
    nec = set(range(2*len(sto)))

    ga = GA(2*len(sto)+1, val, num_gen, num_pop, nec.copy(), purchased.copy(), deliveried.copy(), start, m)
    count = limit_num
    
    st = time.time()
    min_cost = [float('inf')]
    min_path = [start]
    curr_cost = 0
    results = []
    while True :
        nxt = ga.GAP(mat_dist, idx, path, min_path, min_cost, curr_cost)
        if nxt != 'Failure':
            path.append(nxt)
            curr_cost += mat_dist[idx[val[path[-2]]]][idx[val[path[-1]]]]
            if len(path) == len(val):
                c = cost(mat_dist, path, idx, val)
                results.append(c)
                if min_cost[0] > c:
                    min_cost[0] = c
                    min_path = path
                count -= 1
                path = [start]
                curr_cost = 0
                ga = GA(2*len(sto)+1, val, num_gen, num_pop, nec.copy(), purchased.copy(), deliveried.copy(), start, m)
                print('Curr min', c)
                print('NEXT, TRUE')
                continue
        if count < 0 or time.time() - st > limit_time:
            break 
        if nxt == 'Failure':
            path = [start]
            count -= 1
            curr_cost = 0
            ga = GA(2*len(sto)+1, val, num_gen, num_pop, nec.copy(), purchased.copy(), deliveried.copy(), start, m)
            print('NEXT, FALSE')
    path = []
    for i in range(len(min_path)):
        path.append(val[min_path[i]])
    ans = []
    for i in range(len(path)-1):
        part = A_star(G, path[i], path[i+1])
        ans.append(part)
    return min_cost[0], ans, results

#Draw cumulative fuction or probability density mass
def analyze(results, bins=10, cdf=False, pdf=False):
    h = sorted(results)
    if cdf:
        fit = st.norm.cdf(h, np.mean(h), np.std(h))
        pl.plot(h,fit,'-o')
        pl.hist(h, bins=10, density=True, color='purple') 
        pl.xlabel('Results')
        pl.ylabel('Rates')
    if pdf:
        fit = st.norm.pdf(h, np.mean(h), np.std(h))
        pl.plot(h,fit,'-o')
        pl.hist(h, bins=10, density=True, color='purple') 
        pl.xlabel('Results')
        pl.ylabel('Rates')

#Draw graph by function of folium library
def folium_route(G, paths, address, zoom, cus, sto):
    m = folium.Map(location=address, zoom_start=zoom, height=500)
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
    return m
        