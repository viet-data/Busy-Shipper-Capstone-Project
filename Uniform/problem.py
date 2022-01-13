import numpy as np
import pandas as pd
import heapq as hq

class Node:
    def __init__(self, state, path_cost):
        self.state = state
        self.path_cost = path_cost
        self.depth = 0
    
    def __gt__(self, other):
        return True if self.path_cost > other.path_cost else False
    
    def __lt__(self, other):
        return True if self.path_cost < other.path_cost else False
    
    def __ge__(self, other):
        return True if self.path_cost >= other.path_cost else False
    
    def __le__(self, other):
        return True if self.path_cost <= other.path_cost else False
    
    def __ne__(self, other):
        return True if self.path_cost != other.path_cost else False
    
    def __eq__(self, other):
        return True if self.path_cost == other.path_cost else False

class problem:
    def __init__(self,cus, sto, initial_position, orders, G, mat_dist, idx, max_packages=100):
        self.orders = orders
        self.G = G
        self.mat_dist = mat_dist
        self.idx = idx
        self.max_packages = max_packages # max_packages=100
        self.key_block = max([len(str(i)) for i in G.nodes])
        self.cap_block = len(str(max_packages))
        state = '0'*(len(self.orders))*2 + str(initial_position).zfill(self.key_block) + '0'*self.cap_block
        # state: 0101010101_0123456789_000
        self.initial_state = Node(state, 0)
        self.orders.sort(key=lambda x: mat_dist[idx[initial_position]][idx[x[1]]])
        self.cus = []
        self.sto = []
        for i in orders:
            self.cus.append(i[0])
            self.sto.append(i[1])
            
        self.val = {2*i:cus[i] for i in range(len(cus))}
        val2 = {2*i+1:sto[i] for i in range(len(sto))}
        self.val[-2] = initial_position
        self.val.update(val2)
        self.customer = {}
        self.store = {}
        self.k = 0
        idx = 0
        for i in self.orders:
            if i[0] not in self.customer:
                self.customer[i[0]] = [2 * idx]   # id of customer: 0, 2, 4, ...
            else:
                self.customer[i[0]].append(2 * idx)
            
            if i[1] not in self.store:
                self.store[i[1]] = [2 * idx + 1]  # id of store: 1, 3, 5, ...
            else:
                self.store[i[1]].append(2*idx + 1)
            idx += 1  
            
    # Test goal state  
    def goal_test(self, point):
        if point.state[:2*len(self.orders)].count('1') == int(2*len(self.orders)): #state: 111111111_...
            return True
        return False
    
    #Generate the next valid locations
    def next_gen(self, point):
        nxt = []
        for i in range(len(self.orders)*2):
            if point.state[i] == '0':
                nxt.append(i)
        return nxt 
    
    #Generate next children
    def children(self, point):
        nodes = []
        curr_state = point.state
        curr_node = int(point.state[len(self.orders)*2:-self.cap_block])
        curr_pos = curr_state[len(self.orders)*2: len(self.orders)*2 + self.key_block]
        num_pack = int(curr_state[-self.cap_block:])
        nxt = self.next_gen(point)
        for neighbor in nxt:
            cost = point.path_cost + self.mat_dist[self.idx[curr_node]][self.idx[self.val[neighbor]]]
            state = curr_state[:2*len(self.orders)]
            if neighbor % 2 == 0 and state[neighbor+1] == '1':
                state = state[:neighbor] + '1' + state[neighbor+1:]
                state = state + str(self.val[neighbor]).zfill(self.key_block) + str(num_pack - 1).zfill(self.cap_block)
                nodes.append(Node(state, cost))
            if neighbor % 2 == 1 and num_pack < self.max_packages:
                state = state[:neighbor] + '1' + state[neighbor+1:]
                state = state + str(self.val[neighbor]).zfill(self.key_block) + str(num_pack + 1).zfill(self.cap_block)
                nodes.append(Node(state, cost))
        return nodes
