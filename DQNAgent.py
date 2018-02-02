
import tensorflow as tf
import numpy as np

from DQN import DQN


# TODO: train & load implement
class DQNAgent:
    def __init__(self, sess, n_state, action_space, discount_factor, agent_num, epsilon, lr):

        self.sess = sess
        self.n_state = n_state
        self.n_action = len(action_space)
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon['epsilon_start']
        self.epsilon_end = epsilon['epsilon_end']
        self.epsilon_step = epsilon['epsilon_step']
        self.count = 0

        self.pred_network = DQN(self.sess, "pred_network", agent_num, n_state, self.n_action)
        self.target_network = DQN(self.sess, "target_network", agent_num, n_state, self.n_action)
        self.target_network.create_copy_op(self.pred_network)

        self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
        self.action = tf.placeholder(tf.int32, [None], name='action')
        action_onehot = tf.one_hot(self.action, self.n_action, 1.0, 0.0, name='action_onehot')
        pred_q = tf.reduce_sum(self.pred_network.l2 * action_onehot, reduction_indices=1, name='q_acted')
        delta = self.target_q - pred_q
        self.loss = tf.where(tf.abs(delta) < 1.0, 0.5 * tf.square(delta),
                             tf.abs(delta) - 0.5, name='loss')
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=0.95, epsilon=0.1)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        self.update_model = self.trainer.minimize(self.loss)


    def get_action(self, state):
        # decay epsilon value with step count
        epsilon = max(self.epsilon_end, self.epsilon_start - float(self.count) / float(self.epsilon_step))

        state = np.reshape(state, [1, self.n_state])
        # state = state / 100.0  # rescale state
        q_value = self.pred_network.calc_output(state)
        # epsilon = 0.
        if np.random.rand() < epsilon:
            # take random action
            action_idx = np.random.randint(0, self.n_action)
        else:
            # take greedy action
            action_idx = np.argmax(q_value)
        action = float(self.action_space[action_idx])
        self.count += 1
        return action, q_value[0]

    def learn(self, state, action, reward, next_state):
        next_state = np.reshape(next_state, [1, self.n_state])
        # next_state = next_state / 100.0
        state = np.reshape(state, [1, self.n_state])
        # state = state / 100.0

        pred_next_max_action = np.argmax(self.pred_network.calc_output(next_state))

        next_q_value = self.target_network.calc_output(next_state)[0][pred_next_max_action]
        action_idx = self.action_space.index(str(action))
        target_q = reward + self.discount_factor * next_q_value
        _ = self.sess.run(self.update_model, feed_dict={self.pred_network.inputs: state,
                                                        self.target_q: [target_q],
                                                        self.action: [action_idx]})

        if self.count % 100 == 0:
            self.target_network.run_copy()

    def get_greedy_action(self, state):
        state = np.reshape(state, [1, self.n_state])
        # state = state / 100.
        q_value = self.pred_network.calc_output(state)
        action_idx = np.argmax(q_value)
        action = float(self.action_space[action_idx])
        return action




