
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers


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
        save = tf.train.Saver()

        with tf.variable_scope("node_"+str(agent_num)):

            self.inputs = tf.placeholder(tf.float32, [1, n_state], name='inputs')
            l1, l1_w = linear(self.inputs, 32, activation_fn=tf.nn.relu, name='l1')
            self.l2, self.l2_w = linear(l1, self.n_action, activation_fn=None, name='l2')
            self.next_q = tf.placeholder(tf.float32, [1, self.n_action])
            self.loss = tf.reduce_sum(tf.square(self.next_q - self.l2))
            self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
            # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
            self.update_model = self.trainer.minimize(self.loss)

    def get_action(self, state):
        # decay epsilon value with step count
        epsilon = max(self.epsilon_end, self.epsilon_start - float(self.count) / float(self.epsilon_step))

        state = np.reshape(state, [1, self.n_state])
        state = state / 100.0  # rescale state
        q_value = self.sess.run(self.l2, feed_dict={self.inputs: state})

        if np.random.rand() < epsilon:
            # take random action
            action_idx = np.random.choice(self.n_action)
        else:
            # take greedy action
            action_idx = np.argmax(q_value)
        action = float(self.action_space[action_idx])
        self.count += 1
        return action, q_value[0]

    def learn(self, state, action, reward, next_state, q_value):
        next_state = np.reshape(next_state, [1, self.n_state])
        next_state = next_state / 100.0
        state = np.reshape(state, [1, self.n_state])
        state = state / 100.0

        next_q_value = self.sess.run(self.l2, feed_dict={self.inputs: next_state})
        action_idx = self.action_space.index(str(action))
        target_q = q_value
        target_q[action_idx] = reward + self.discount_factor * np.max(next_q_value)
        target_q = np.reshape(target_q, [1, self.n_action])
        _ = self.sess.run(self.update_model, feed_dict={self.inputs: state, self.next_q: target_q})

    def get_greedy_action(self, state):
        state = np.reshape(state, [1, self.n_state])
        state = state / 100.
        q_value = self.sess.run(self.l2, feed_dict={self.inputs: state})
        action_idx = np.argmax(q_value)
        action = float(self.action_space[action_idx])
        return action


# fully-connected layer for neural network
def linear(input_,
           output_size,
           weights_initializer=initializers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=None,
           trainable=True,
           name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                            initializer=weights_initializer, trainable=trainable)
        # b = tf.get_variable('b', [output_size],
        #                    initializer=biases_initializer, trainable=trainable)
        # out = tf.nn.bias_add(tf.matmul(input_, w), b)
        out = tf.matmul(input_, w)
    if activation_fn is not None:
        return activation_fn(out), w  #, b
    else:
        return out,  w  #, b

