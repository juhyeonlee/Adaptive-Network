
import numpy as np
import networkx as nx


class Environment:
    def __init__(self, dims, mu, init_txr):
        self.one_dim = dims
        self.max_block = self.one_dim ** 2
        self.mu = mu # node density
        self.beta = 0.0
        self.txr = init_txr
        self.last_txr = init_txr
        self.action_space = [-3, -2, -1, 0, 1, 2, 3]
        self.n_actions = len(self.action_space)

        # node state
        self.block_coords = np.zeros((self.max_block, 2))
        self.no_events = np.zeros(self.max_block, dtype=np.int32)
        self.coords = []
        self.players = []
        self.node_loc = []
        self.len_players = 0
        self.num_node = 0

        # reward
        self.connectivity_ratio = 0.
        self.goodput = 0.
        self.reward = 0.
        self.utility_coeff = 0.53

        # default coordination for each block
        i = 0
        for c2 in range(0, 2 * self.one_dim - 1, 2):
            for c1 in range(0, 2 * self.one_dim - 1, 2):
                self.block_coords[i, :] = [c1, c2]
                i += 1

    def step(self, action):
        self.txr = self.last_txr + action
        self.last_txr = self.txr
        d = np.zeros((self.num_node, self.num_node))
        for f in range(self.num_node):
            for t in range(self.num_node):
                d[f, t] = np.linalg.norm(np.array(self.node_loc[f]) - np.array(self.node_loc[t]))

        # adjacent matrix
        adj_matrix = np.zeros((self.num_node, self.num_node))
        pp = self.players + list(range(self.num_node - 4, self.num_node))
        for p in pp:
            adj_matrix[p] = ((d[p] <= self.txr) * np.random.rand(1)) > self.beta

        G = nx.DiGraph()
        for i in range(self.num_node):
            G.add_node(i)
            for j in range(self.num_node):
                if adj_matrix[i, j] == 1:
                    G.add_edges_from([(i, j)])

        # find shortest path between sources and destinations (2 * 2)
        dist = []
        path = []
        for s in [self.num_node - 4, self.num_node - 3]:  # source
            for t in [self.num_node - 2, self.num_node - 1]:  # destination
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
        self.connectivity_ratio = 0.
        for i in range(4):
            if dist[i] is not np.inf:
                self.connectivity_ratio += 1.
        self.connectivity_ratio /= 4.
        self.goodput = np.sum(np.divide(1, dist))
        print("connectivity ratio: ", self.connectivity_ratio)
        print("goodput: ", self.goodput)
        self.reward = self.goodput * self.utility_coeff - action * (1.0 - self.utility_coeff)

        # next state
        # node state
        self.no_events = np.zeros(self.max_block, dtype=np.int32)
        self.coords = []
        # block coordination random generation
        for block in range(self.max_block):
            # the number of nodes per block based on poisson point process
            self.no_events[block] = int(np.random.poisson(self.mu * 4))
            # random location
            self.coords.append(2 * np.random.rand(self.no_events[block], 2))

        self.players = []
        # the number of players
        for t in range(1, self.one_dim - 1):
            self.players += [i for i in
                             range(np.sum(self.no_events[0:t * self.one_dim + 1]) + 1,
                                   np.sum(self.no_events[0:(t + 1) * self.one_dim - 1]) + 1)]

        self.len_players = len(self.players)
        print("num of players: ", self.len_players)

        self.node_loc = []
        for b in range(self.max_block):
            for n in range(self.no_events[b]):
                self.node_loc.append(self.block_coords[b] + self.coords[b][n])

        # two source nodes: fixed location
        self.node_loc.append([2, 2 * (self.one_dim - 1)])
        self.node_loc.append([2 * (self.one_dim - 1), 2 * (self.one_dim - 1)])

        # two destination nodes: fixed location
        self.node_loc.append([2, 2])
        self.node_loc.append([2 * (self.one_dim - 1), 2])

        # distance matrix
        self.num_node = len(self.node_loc)
        print("num of nodes: ", self.num_node)

        return self.num_node, self.reward

    def reset(self, beta, init_txr):
        self.beta = beta
        self.txr = init_txr
        self.last_txr = init_txr

        # node state
        self.no_events = np.zeros(self.max_block, dtype=np.int32)
        self.coords = []
        self.players = []
        self.node_loc = []

        # reward
        self.connectivity_ratio = 0.
        self.goodput = 0.
        self.reward = 0.

        # block coordination random generation
        for block in range(self.max_block):
            # the number of nodes per block based on poisson point process
            self.no_events[block] = int(np.random.poisson(self.mu * 4))
            # random location
            self.coords.append(2 * np.random.rand(self.no_events[block], 2))

        # the number of players
        for t in range(1, self.one_dim - 1):
            self.players += [i for i in
                             range(np.sum(self.no_events[0:t * self.one_dim + 1]) + 1,
                                   np.sum(self.no_events[0:(t + 1) * self.one_dim - 1]) + 1)]

        self.len_players = len(self.players)
        print("num of players: ", self.len_players)

        for b in range(self.max_block):
            for n in range(self.no_events[b]):
                self.node_loc.append(self.block_coords[b] + self.coords[b][n])

        # two source nodes: fixed location
        self.node_loc.append([2, 2 * (self.one_dim - 1)])
        self.node_loc.append([2 * (self.one_dim - 1), 2 * (self.one_dim - 1)])

        # two destination nodes: fixed location
        self.node_loc.append([2, 2])
        self.node_loc.append([2 * (self.one_dim - 1), 2])

        # distance matrix
        self.num_node = len(self.node_loc)
        print("num of nodes: ", self.num_node)
        return self.num_node


