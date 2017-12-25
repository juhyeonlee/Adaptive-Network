
"""
adaptive network
2017-11-13
written by minhae kwon
python vers. ported by juhyeon lee
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
1. parameter setting
2. initial network formation
   - each block size is 2 by 2
   - consider (5 by 5) blocks for one network
   - [ x x x x x ]
   - [ x 7 8 9 x ] 
   - [ x 4 5 6 x ]
   - [ x 1 2 3 x ]
   - [ x x x x x ]
   - only 3 by 3 blocks are used for data delivery
   - 2 source nodes (at block 7 & 9) and 2 destination nodes (at block 1 & 3)-fixed location
   - number of nodes per block: based on poisson point process
   - location of nodes: random generation inside of each block
 3. evaluate network performance
   - connectivity ratio
   - goodput
"""

# 1. parameter setting
mu = 4/5    # node density
one_dim = 5
max_block = one_dim ** 2
beta = 0.1  # link failure rate
tx_r = 2    # default radius of tx range # action

# 2. initialize network formation
no_events = np.zeros(max_block, dtype=np.int32)
coords = []

# default coordination for each block
block_coords = np.zeros((max_block, 2))
i = 0
for c2 in range(0, 2*one_dim - 1, 2):
    for c1 in range(0, 2*one_dim - 1, 2):
        block_coords[i, :] = [c1, c2]
        i += 1
# block coordination random generation
for block in range(max_block):
    # the number of nodes per block based on poisson point process
    no_events[block] = int(np.random.poisson(mu * 4))
    # random location
    coords.append(2 * np.random.rand(no_events[block], 2))

# the number of players
players = []
for t in range(1, one_dim - 1):
    players += [i for i in
                    range(np.sum(no_events[0:t*one_dim+1])+1,
                          np.sum(no_events[0:(t+1)*one_dim-1])+1)]

len_players = len(players)
print("num of players: ", len_players)

# node location
node_loc = []
for b in range(max_block):
    for n in range(no_events[b]):
        node_loc.append(block_coords[b] + coords[b][n])


# two source nodes: fixed location
node_loc.append([2, 2 * (one_dim - 1)])
node_loc.append([2 * (one_dim - 1), 2 * (one_dim - 1)])

# two destination nodes: fixed location
node_loc.append([2, 2])
node_loc.append([2*(one_dim - 1), 2])

# distance matrix
num_node = len(node_loc)
print("num of nodes: ", num_node)
d = np.zeros((num_node, num_node))
for f in range(num_node):
    for t in range(num_node):
        d[f, t] = np.linalg.norm(np.array(node_loc[f]) - np.array(node_loc[t]))

# adjacent matrix
adj_matrix = np.zeros((num_node, num_node))
pp = players + list(range(num_node-4, num_node))
# TODO: same node --> same random number?
for p in pp:
    adj_matrix[p] = ((d[p] <= tx_r) * np.random.rand(1)) > beta

# 3. evaluate network performance
# plot network
G = nx.DiGraph()
for i in range(num_node):
    G.add_node(i)
    for j in range(num_node):
        if adj_matrix[i,j] == 1:
            G.add_edges_from([(i,j)])
# red coloring for source and destination nodes
val_map = {num_node - 1: 0.57,
           num_node - 2: 0.57,
           num_node - 3: 0.57,
           num_node - 4: 0.57}
values = [val_map.get(node, 0.1) for node in G.nodes()]
labels = {}
for node in G.nodes():
    labels[node] = str(node)
nx.draw_networkx_nodes(G, node_loc, cmap=plt.get_cmap('jet'),
                       node_color=values, node_size=10)
nx.draw_networkx_labels(G, node_loc, labels=labels, font_size=8)
nx.draw_networkx_edges(G, node_loc, edge_color='c', arrows=True)
plt.grid()
# plt.show()

# find shortest path between sources and destinations (2 * 2)
dist = []
path = []
for s in [num_node-4, num_node-3]: # source
    for t in [num_node-2, num_node-1]: # destination
        if nx.has_path(G, s, t):
            p_ = nx.shortest_path(G, s, t)
            path.append(p_)
            dist.append(len(p_))
        else:
            path.append([])
            dist.append(np.inf)

print("path between sources and destinations: ", path)
print("distance: ", dist)

# calculate connectivity ratio and goodput
# connectivity ratio: the number of connected path, max 1.0 (source 2 * dest 2 ) / 4
connectivity_ratio = 0.
for i in range(4):
    if dist[i] is not np.inf:
        connectivity_ratio += 1.
connectivity_ratio /= 4.
goodput = np.sum(np.divide(1, dist))
print("connectivity ratio: ", connectivity_ratio)
print("goodput: ", goodput)
