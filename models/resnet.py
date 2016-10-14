"""basic residual network class."""

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import fully_connected

from models import basic_resnet
from models.basic_resnet import UnitsGroup
# from utils import logger
from utils import FLAGS

FLAGS.add('--groups_conf',
          type=str,
          default=''
          '3, 16, 32, 0\n'
          '4, 32, 64, 1\n'
          '6, 64, 128, 1\n'
          '3, 128, 256, 0',
          help='Configurations of different residual groups.')
FLAGS.add('--max_pool',
          type=bool,
          default=True,
          help='whether to add max pooling before first residual unit.')


class Model(basic_resnet.Model):
    """Residual neural network model.
    classify web page only based on target html."""

    def resnn(self, image_batch):
        """Build the resnn model.
        Args:
            image_batch: Sequences returned from inputs_train() or inputs_eval.
        Returns:
            Logits.
        """
        # First convolution
        with tf.variable_scope('conv_layer1'):
            net = self.conv2d(image_batch, self.groups[0].num_ker, 5, 1)
            net = self.BN_ReLU(net)

        # Max pool
        if FLAGS.max_pool:
            net = tf.nn.max_pool(net,
                                 [1, 3, 3, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')

        # stacking Residual Units
        for group_i, group in enumerate(self.groups):
            for unit_i in range(group.num_units):
                net = self.residual_unit(net, group_i, unit_i)

        # an extra activation before average pooling
        if FLAGS.special_first:
            with tf.variable_scope('special_BN_ReLU'):
                net = self.BN_ReLU(net)

        # padding should be VALID for global average pooling
        # output: batch*1*1*channels
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(net,
                             ksize=[1, net_shape[1], net_shape[2], 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID')

        net_shape = net.get_shape().as_list()
        softmax_len = net_shape[1] * net_shape[2] * net_shape[3]
        net = tf.reshape(net, [-1, softmax_len])

        # add dropout
        if FLAGS.dropout:
            with tf.name_scope("dropout"):
                net = tf.nn.dropout(net, FLAGS.dropout_keep_prob)

        # 2D-fully connected nueral network
        with tf.variable_scope('FC-layer'):
            net = fully_connected(
                net,
                num_outputs=FLAGS.num_cats,
                activation_fn=None,
                normalizer_fn=None,
                weights_initializer=variance_scaling_initializer(),
                weights_regularizer=l2_regularizer(FLAGS.weight_decay),
                biases_initializer=tf.zeros_initializer, )

        return net
