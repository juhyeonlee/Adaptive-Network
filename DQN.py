
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

class DQN:
    def __init__(self, sess, name, agent_num, n_state, n_action):

        self.sess = sess
        self.var = {}
        self.name = name + "_" + str(agent_num)
        self.copy_op = None
        with tf.variable_scope(self.name):
            self.inputs = tf.placeholder(tf.float32, [1, n_state], name='inputs')
            self.l1, self.var['l1_w'] = linear(self.inputs, 32, activation_fn=tf.nn.relu, name='l1')
            self.l2, self.var['l2_w'] = linear(self.l1, n_action, activation_fn=None, name='l2')

    def calc_output(self, state):
        return self.sess.run(self.l2, feed_dict={self.inputs: state})

    def create_copy_op(self, network):
        with tf.variable_scope(self.name):
            copy_ops = []

            for name in self.var.keys():
                copy_op = self.var[name].assign(network.var[name])
                copy_ops.append(copy_op)

            self.copy_op = tf.group(*copy_ops, name='copy_op')

    def run_copy(self):
        if self.copy_op is None:
            raise Exception("run 'create_copy_op' first before copy")
        else:
            self.sess.run(self.copy_op)

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