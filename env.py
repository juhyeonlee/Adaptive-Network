
import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt

class AdhocNetEnv:
    def __init__(self, action_space, args):
        self.nw_size = args['nw_size']
        self.num_blocks = args['nw_size'] ** 2
        self.mu = args['mu'] # node density
        self.init_txr = args['init_txr']
        self.source_init_txr = args['source_init_txr']
        self.beta = args['beta'] # noise
        self.action_sapce = action_space

        # reward parameters
        self.utility_coeff = args['utility_coeff'] # weight on goodput
        self.utility_pos_coeff = args['utility_pos_coeff'] # to make utiltiy to be positive

        # node state
        # TODO: !!!! 다 바꿔버리기 state = loc + player 상태 --> obs 추출 따로 만들기
        self.block_coords = np.zeros((self.num_blocks, 2))
        self.num_agents_in_blocks = np.zeros(self.num_blocks, dtype=np.int32)
        self.players = []
        self.node_loc = []
        self.num_players = 0
        self.num_node = 0
        self.d = None
        self.beta_matrix = None
        self.last_goodput = 0.0
        self.last_txr = None

        # default coordination for each block
        i = 0
        for c2 in range(0, 2 * self.nw_size - 1, 2):
            for c1 in range(0, 2 * self.nw_size - 1, 2):
                self.block_coords[i, :] = [c1, c2]
                i += 1

    def step(self, actions, steps, ep):
        actions_txr = self.action_sapce[actions.astype('int')]
        txr = self.last_txr + actions_txr
        # txr is between 0 and 3
        txr = np.clip(txr, 0, 3)
        #print(txr)
        # source node tx_r is fixed as s_init_txr
        txr[len(txr) - 4] = self.source_init_txr
        txr[len(txr) - 3] = self.source_init_txr
        self.last_txr = txr
        # print("action: ", action)
        # print("updated TX range: ", txr)
        energy = txr ** 2

        # adjacent matrix
        adj_matrix = np.zeros((self.num_node, self.num_node))
        players_all = self.players + list(range(self.num_node - 4, self.num_node))
        for idx, p in enumerate(players_all):
            adj_matrix[p] = (self.d[p] <= float(txr[idx])) * self.beta_matrix[p]
        for i in range(self.num_node):
            adj_matrix[i, i] = 1

        G = nx.DiGraph()
        for i in range(self.num_node):
            G.add_node(i)
            for j in range(self.num_node):
                if adj_matrix[i, j] == 1:
                    G.add_edges_from([(i, j)])
        # red coloring for source and destination nodes

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

        # print("path between sources and destinations: ", path)
        # print("distance: ", dist)
        #
        # plt.clf()
        # if steps % 30 == 0:
        #     val_map = {self.num_node - 1: 0.57,
        #                self.num_node - 2: 0.57,
        #                self.num_node - 3: 0.57,
        #                self.num_node - 4: 0.57}
        #     values = [val_map.get(node, 0.1) for node in G.nodes()]
        #     labels = {}
        #     for node in G.nodes():
        #         labels[node] = str(node)
        #     nx.draw_networkx_nodes(G, self.node_loc, cmap=plt.get_cmap('jet'),
        #                            node_color=values, node_size=10)
        #     nx.draw_networkx_labels(G, self.node_loc, labels=labels, font_size=8)
        #     nx.draw_networkx_edges(G, self.node_loc, edge_color='c', arrows=True)
        #     for p in path:
        #         if len(p) is not 0:
        #             path_edges = zip(p, p[1:])
        #             # print(list(path_edges))
        #             nx.draw_networkx_nodes(G, self.node_loc,  nodelist=p, node_color='r', node_size=10)
        #             nx.draw_networkx_edges(G, self.node_loc, edgelist=list(path_edges), edge_color='r', arrows=True)
        #
        #     # for idx, p in enumerate(pp):
        #     #     if self.txr[idx] > 0.5:
        #     #         circle = plt.Circle(self.node_loc[p], self.txr[idx], edgecolor='r', facecolor='none')
        #     #         plt.gca().add_patch(circle)
        #
        #     plt.grid()
        #     plt.figure(10)
            # plt.savefig('fig1_one_trial_network' + str(steps) + '_' + str(ep)+'.eps')
        #

        # calculate connectivity ratio and goodput
        # connectivity ratio:
        #  the number of connected path, max 1.0 (source 2 * dest 2 ) / 4
        connectivity_ratio = 0.
        for i in range(4):
            if dist[i] is not np.inf:
                connectivity_ratio += 1.
        connectivity_ratio /= 4.
        goodput = np.sum(np.divide(1., dist))
        # print("connectivity ratio: ", connectivity_ratio)
        # print("goodput: ", goodput)
        # print("=======================")
        # TODO: constant for positive value
        # reward = goodput  - actions_txr * (0.4)
        # reward = self.utility_pos_coeff +connectivity_ratio * ( goodput * self.utility_coeff - actions_txr)
        # reward = self.utility_pos_coeff + goodput * self.utility_coeff - actions_txr
        # reward = (self.utility_pos_coeff + 2 * connectivity_ratio + (goodput * self.utility_coeff - actions_txr))
        # reward = self.utility_pos_coeff + self.utility_coeff * (goodput - self.last_goodput) - actions_txr
        reward = self.utility_pos_coeff + self.utility_coeff * 20 * (goodput - self.last_goodput) - (1-self.utility_coeff) * actions_txr

        # print('goodput improvement ',  goodput - self.last_goodput)
        # print('action', action)
        # print("reward: ", reward)
        reward = reward / 10.  # rescaling reward to train NN stable
        # print("reward: ", reward)
        # next state
        # TODO: only change node location
        # change node location
        agents_coords = []
        for block in range(self.num_blocks):
            # random location
            agents_coords.append(2 * np.random.rand(self.num_agents_in_blocks[block], 2))

        self.node_loc = []
        for b in range(self.num_blocks):
            for n in range(self.num_agents_in_blocks[b]):
                self.node_loc.append(self.block_coords[b] + agents_coords[b][n])

        # two source nodes: fixed location
        self.node_loc.append([2, 2 * (self.nw_size - 1)])
        self.node_loc.append([2 * (self.nw_size - 1), 2 * (self.nw_size - 1)])

        # two destination nodes: fixed location
        self.node_loc.append([2, 2])
        self.node_loc.append([2 * (self.nw_size - 1), 2])

        # # distance matrix
        # self.num_node = len(self.node_loc)
        # #print("num of nodes: ", self.num_node)

        self.d = np.zeros((self.num_node, self.num_node))
        for f in range(self.num_node):
            for t in range(self.num_node):
                self.d[f, t] = np.linalg.norm(np.array(self.node_loc[f]) - np.array(self.node_loc[t]))

        # adjacent matrix
        adj_matrix = np.zeros((self.num_node, self.num_node))
        self.beta_matrix = np.random.rand(self.num_node, self.num_node) > self.beta
        players_all = self.players + list(range(self.num_node - 4, self.num_node))
        for idx, p in enumerate(players_all):
            adj_matrix[p] = (self.d[p] <= float(txr[idx])) * self.beta_matrix[p]
        for i in range(self.num_node):
            adj_matrix[i, i] = 1

        # TODO: including source and terminal nodes??
        current_state = np.zeros(self.num_players + 4)
        for idx, p in enumerate(players_all):
            current_state[idx] = np.sum(adj_matrix[p]) / 100.
            # self.current_state[i][1] = self.txr[i] / 6.0

        self.last_goodput = goodput

        return current_state, reward, goodput, energy, connectivity_ratio

    def reset(self):
        # node state
        self.num_agents_in_blocks = np.zeros(self.num_blocks, dtype=np.int32)
        agents_coords = []
        self.players = []
        self.node_loc = []

        # block coordination random generation
        for block_idx in range(self.num_blocks):
            # the number of nodes per block based on poisson point process
            self.num_agents_in_blocks[block_idx] = int(np.random.poisson(self.mu * 4))
            # random location
            agents_coords.append(2 * np.random.rand(self.num_agents_in_blocks[block_idx], 2))

        # the number of players (plyaers: the nodes between sources and destinations)
        for t in range(1, self.nw_size - 1):
            self.players += [i for i in
                             range(int(np.sum(self.num_agents_in_blocks[:t * self.nw_size + 1])),
                                   int(np.sum(self.num_agents_in_blocks[:(t + 1) * self.nw_size - 1])))]

        self.num_players = len(self.players)
        # print("num of players: ", self.num_players)
        print("players index: ", self.players)

        for b in range(self.num_blocks):
            for n in range(self.num_agents_in_blocks[b]):
                self.node_loc.append(self.block_coords[b] + agents_coords[b][n])

        # two source nodes: fixed location
        self.node_loc.append([2, 2 * (self.nw_size - 1)])
        self.node_loc.append([2 * (self.nw_size - 1), 2 * (self.nw_size - 1)])

        # two destination nodes: fixed location
        self.node_loc.append([2, 2])
        self.node_loc.append([2 * (self.nw_size - 1), 2])

        # distance matrix
        self.num_node = len(self.node_loc)
        # print("num of nodes: ", self.num_node)

        self.d = np.zeros((self.num_node, self.num_node))
        for f in range(self.num_node):
            for t in range(self.num_node):
                self.d[f, t] = np.linalg.norm(np.array(self.node_loc[f]) - np.array(self.node_loc[t]))

        # adjacent matrix
        adj_matrix = np.zeros((self.num_node, self.num_node))
        self.beta_matrix = np.random.rand(self.num_node, self.num_node) > self.beta
        # players_all : players + source nodes + destination nodes
        players_all = self.players + list(range(self.num_node - 4, self.num_node))

        txr = np.ones((self.num_players + 4), dtype=np.int32) * int(self.init_txr)
        txr[len(txr) - 4] = int(self.source_init_txr)
        txr[len(txr) - 3] = int(self.source_init_txr)

        # for p in pp:
        #     # for source nodes
        #     if p == self.num_node-4 :
        #         self.adj_matrix[p] = (self.d[p] <= float(self.s_init_txr)) * self.beta_matrix[p]
        #     elif p == self.num_node - 3:
        #         self.adj_matrix[p] = (self.d[p] <= float(self.s_init_txr)) * self.beta_matrix[p]
        #     # for agents (intermediate nodes)
        #     else:
        #         self.adj_matrix[p] = (self.d[p] <= float(self.init_txr)) * self.beta_matrix[p]
        for idx, p in enumerate(players_all):
            adj_matrix[p] = (self.d[p] <= float(txr[idx])) * self.beta_matrix[p]

        for i in range(self.num_node):
            adj_matrix[i, i] = 1

        current_state = np.zeros(self.num_players + 4)
        for idx, p in enumerate(players_all):
            current_state[idx] = np.sum(adj_matrix[p]) / 100.

        # save current transmission range
        self.last_txr = txr
        self.last_goodput = 0.0
        return current_state


