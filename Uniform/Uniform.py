from problem import *
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

#return soluion for A*
def solution2(parents, des):
    ans = []
    while des in parents:
        ans.append(des)
        des = parents[des]
    ans.append(des)
    return ans[::-1]

#A* find the shortest path between two given points
def A_star(G, ori, des):
    frontier = [(0,ori)]
    checked = {ori:0}
    parents = {}
    while frontier:
        est, pos = hq.heappop(frontier)
        if pos == des:
            return solution2(parents, des)
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

#return cost of path
def cost(mat_dist, path, idx):
    c = 0
    for i in range(0, len(path)-1):
        c += mat_dist[idx[path[i]]][idx[path[i+1]]]
    return c

#return solution for uniform-cost search
def solution(point, parents, problem):
    ans = []
    point = point.state
    s = len(problem.orders)*2
    e = len(problem.orders)*2 + problem.key_block
    while point in parents:
        ans.append(int(point[s:e]))
        point = parents[point]
    ans.append(int(point[s:e]))
    return ans[::-1]

#Uniform cost search
def uniform_cost_search(problem):
    node = problem.initial_state
    frontier = [node]
    checked = {}
    parents = {}
    while frontier:
        node = hq.heappop(frontier)
        if problem.goal_test(node):
            path = solution(node, parents, problem)
            ans = []
            for i in range(len(path)-1):
                part = A_star(problem.G, path[i], path[i+1])
                ans.append(part)
            c = cost(problem.mat_dist, path, problem.idx)
            return ans, c
        checked[node.state] = node.path_cost
        children = problem.children(node)
        for child in children:
            if child.state not in checked or child.path_cost < checked[child.state]:
                checked[child.state] = child.path_cost
                parents[child.state] = node.state
                hq.heappush(frontier, child)
    return "Impossible", "Impossible"


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
                folium.Marker([y,x], popup='Store', icon=folium.Icon(color="red")).add_to(m)
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