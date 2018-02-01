
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt


class Environment:
    def __init__(self, dims, mu, init_txr, utility_coeff, utility_pos_coeff, action_space):
        self.one_dim = dims
        self.max_block = self.one_dim ** 2
        self.mu = mu # node density
        self.beta = 0.0
        self.init_txr = init_txr
        self.txr = None
        #self.action_space = [-3, -2, -1, 0, 1, 2, 3]
        self.action_space = action_space
        self.n_actions = len(self.action_space)

        # node state
        self.block_coords = np.zeros((self.max_block, 2))
        self.no_events = np.zeros(self.max_block, dtype=np.int32)
        self.coords = []
        self.players = []
        self.node_loc = []
        self.num_players = 0
        self.num_node = 0
        self.d = None
        self.adj_matrix = None
        self.current_state = None
        self.beta_matrix = None

        # reward parameters
        self.utility_coeff = utility_coeff # weight on goodput
        self.utility_pos_coeff = utility_pos_coeff # to make utiltiy to be positive

        # default coordination for each block
        i = 0
        for c2 in range(0, 2 * self.one_dim - 1, 2):
            for c1 in range(0, 2 * self.one_dim - 1, 2):
                self.block_coords[i, :] = [c1, c2]
                i += 1

    def step(self, action, beta):
        self.beta = beta
        self.txr = self.last_txr + action
        # txr is between 0 and 6
        for i in range(len(self.txr)):
            if self.txr[i] < 0:
                self.txr[i] = 0
            elif self.txr[i] > 5:
                self.txr[i] = 5
        self.last_txr = self.txr
        print("action: ", action)
        print("updated TX range: ", self.txr)
        energy = self.txr ** 2

        # adjacent matrix
        self.adj_matrix = np.zeros((self.num_node, self.num_node))
        pp = self.players + list(range(self.num_node - 4, self.num_node))
        for idx, p in enumerate(pp):
            self.adj_matrix[p] = (self.d[p] <= float(self.txr[idx])) * self.beta_matrix[p]
        for i in range(self.num_node):
            self.adj_matrix[i, i] = 1

        G = nx.DiGraph()
        for i in range(self.num_node):
            G.add_node(i)
            for j in range(self.num_node):
                if self.adj_matrix[i, j] == 1:
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
        connectivity_ratio = 0.
        for i in range(4):
            if dist[i] is not np.inf:
                connectivity_ratio += 1.
        connectivity_ratio /= 4.
        goodput = np.sum(np.divide(1., dist))
        print("connectivity ratio: ", connectivity_ratio)
        print("goodput: ", goodput)
        print("=======================")
        #TODO: constant for positive value
        # reward = goodput  - action * (0.4)
        # reward = self.utility_pos_coeff +connectivity_ratio * ( goodput * self.utility_coeff - action)
        reward = self.utility_pos_coeff + goodput * self.utility_coeff - action
        #reward = (self.utility_pos_coeff + 2 * connectivity_ratio + (goodput * self.utility_coeff - action))
        reward = reward / 10.  # rescaling reward to train NN stable
        print("reward: ", reward)
        # next state
        #TODO: only change node location
        # change node location
        self.coords = []
        for block in range(self.max_block):
            # random location
            self.coords.append(2 * np.random.rand(self.no_events[block], 2))

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
        #print("num of nodes: ", self.num_node)

        self.d = np.zeros((self.num_node, self.num_node))
        for f in range(self.num_node):
            for t in range(self.num_node):
                self.d[f, t] = np.linalg.norm(np.array(self.node_loc[f]) - np.array(self.node_loc[t]))

        # adjacent matrix
        self.adj_matrix = np.zeros((self.num_node, self.num_node))
        self.beta_matrix = np.random.rand(self.num_node, self.num_node) > self.beta
        pp = self.players + list(range(self.num_node - 4, self.num_node))
        for idx, p in enumerate(pp):
            self.adj_matrix[p] = (self.d[p] <= float(self.txr[idx])) * self.beta_matrix[p]
        for i in range(self.num_node):
            self.adj_matrix[i, i] = 1

        # TODO: including source and terminal nodes??
        self.current_state = np.zeros(self.num_players + 4)
        for i in range(self.num_players + 4):
            self.current_state[i] = np.sum(self.adj_matrix[pp[i]]) / 100.
            # self.current_state[i][1] = self.txr[i] / 6.0

        return self.current_state, reward, goodput, energy

    def reset(self, init_txr):
        self.beta = 0
        self.init_txr = init_txr

        # node state
        self.no_events = np.zeros(self.max_block, dtype=np.int32)
        self.coords = []
        self.players = []
        self.node_loc = []

        # block coordination random generation
        for block in range(self.max_block):
            # the number of nodes per block based on poisson point process
            self.no_events[block] = int(np.random.poisson(self.mu * 4))
            # random location
            self.coords.append(2 * np.random.rand(self.no_events[block], 2))

        # the number of players
        for t in range(1, self.one_dim - 1):
            self.players += [i for i in
                             range(np.sum(self.no_events[0:t * self.one_dim + 1]),
                                   np.sum(self.no_events[0:(t + 1) * self.one_dim - 1]))]

        self.num_players = len(self.players)
        print("num of players: ", self.num_players)
        print("players index: ", self.players)

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

        self.d = np.zeros((self.num_node, self.num_node))
        for f in range(self.num_node):
            for t in range(self.num_node):
                self.d[f, t] = np.linalg.norm(np.array(self.node_loc[f]) - np.array(self.node_loc[t]))

        # adjacent matrix
        self.adj_matrix = np.zeros((self.num_node, self.num_node))
        self.beta_matrix = np.random.rand(self.num_node, self.num_node) > self.beta
        pp = self.players + list(range(self.num_node - 4, self.num_node))
        for p in pp:
            self.adj_matrix[p] = (self.d[p] <= float(self.init_txr)) * self.beta_matrix[p]
        for i in range(self.num_node):
            self.adj_matrix[i, i] = 1

        #TODO: including source and terminal nodes??
        self.current_state = np.zeros(self.num_players + 4)
        for i in range(self.num_players + 4):
            self.current_state[i] = np.sum(self.adj_matrix[pp[i]]) / 100.
            # self.current_state[i][1] = self.init_txr / 6.0

        # save current transmission range
        self.last_txr = np.ones((self.num_players + 4), dtype=np.int32) * self.init_txr

        return self.current_state


