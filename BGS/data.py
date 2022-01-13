from BGS import cutdown

# read input data
def input(filename):
    with open(filename) as f:
        s, m = map(int, f.readline().split())
        n = int(f.readline())
        orders = []
        for i in range(n):
            cus, sto = map(int, f.readline().split())
            orders.append((cus, sto))
    return s, m, n, orders

# cut down problem to problem on map contain only locations that are store, customer or initial positon.
def cut_down(G, start, orders):
    cus = []
    sto = []
    for i in orders[:]:
        cus.append(i[0])
        sto.append(i[1])
    lst, std, nxt = cutdown.arrange(sto + [start], cus + [start], G, graph=False)
    mat_dist = cutdown.getdist(lst, std, nxt, G)
    idx = {lst[i]: i for i in range(len(lst))}
    return mat_dist, lst, idx, cus, sto

# return all necessary data for algorithm
def get_data(filename, G):
    s, m, n, orders= input(filename)
    mat_dist, lst, idx, cus, sto = cut_down(G, s, orders)
    return s, m, mat_dist, lst, idx, orders, cus, sto