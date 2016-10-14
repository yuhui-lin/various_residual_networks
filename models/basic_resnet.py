"""basic residual network class."""
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib.layers import convolution2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers.python.layers import utils

from models import model
# from utils import logger
from utils import FLAGS

FLAGS.add('--special_first',
          type=bool,
          default=True,
          help='special first residual unit from P14 of '
          '(arxiv.org/abs/1603.05027')
FLAGS.add('--shortcut',
          type=int,
          default=1,
          help='shortcut connection type: (arXiv:1512.03385)'
          '0: 0-padding and average pooling'
          '1: convolution projection only for increasing dimension'
          '2: projection for all shortcut')
FLAGS.add('--unit_type',
          type=int,
          default=0,
          help='# the type of residual unit '
          '# 0: post-activation; 1: pre-activation')
FLAGS.add('--residual_type',
          type=int,
          default=1,
          help='# residual function: 0: bottleneck'
          '# 1: basic two conv')
FLAGS.add('--bott_size_mid',
          type=int,
          default=3,
          help='the middle conv window size of bottleneck: 3, 4, 5')
FLAGS.add('--bott_size_ends',
          type=int,
          default=1,
          help='window size of first and third conv in bottleneck')
FLAGS.add('--dropout',
          type=bool,
          default=True,
          help='whether enable dropout before FC layer')
# FLAGS.add('--if_drop ',
#           type=bool,
#           default=False,
#           help='whehter use dropout in residual function')
FLAGS.add('--wide_factor',
          type=int,
          default=1,
          help='widening factor k of wide residual network.')

# Configurations for each group
# several residual units (aka. bottleneck blocks) form a group
# no more than three groups with downsampling
UnitsGroup = namedtuple(
    'UnitsGroup',
    [
        'num_units',  # number of residual units
        'num_ker',  # number of kernels for each convolution
        'num_key_exp',  # number of expanded kernels
        'is_downsample'  # (int): downsample data using stride 2
        # types of BottleneckBlock ??
        # wide resnet kernel*k ??
    ])


class Model(model.Model):
    """Residual neural network model.
    classify web page only based on target html."""

    def BN_ReLU(self, net):
        """Batch Normalization and ReLU."""
        # 'gamma' is not used as the next layer is ReLU
        net = batch_norm(net,
                         center=True,
                         scale=False,
                         activation_fn=tf.nn.relu, )
        self._activation_summary(net)
        return net

        # def conv2d(self, net, num_ker, ker_size, stride):
        # 1D-convolution
        net = convolution2d(
            net,
            num_outputs=num_ker,
            kernel_size=[ker_size, 1],
            stride=[stride, 1],
            padding='SAME',
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=variance_scaling_initializer(),
            weights_regularizer=l2_regularizer(self.weight_decay),
            biases_initializer=tf.zeros_initializer)
        return net

    def conv2d(self, net, num_ker, ker_size, stride):
        net = convolution2d(
            net,
            num_outputs=num_ker,
            kernel_size=[ker_size, ker_size],
            stride=[stride, stride],
            padding='SAME',
            activation_fn=None,
            normalizer_fn=None,
            weights_initializer=variance_scaling_initializer(),
            weights_regularizer=l2_regularizer(FLAGS.weight_decay),
            biases_initializer=tf.zeros_initializer)
        return net

    def conv_pre(self, name, net, num_ker, kernel_size, stride, is_first):
        """ 1D pre-activation convolution.
        args:
            num_ker (int): number of kernels (out_channels).
            ker_size (int): size of 1D kernel.
            stride (int)
        """
        with tf.variable_scope(name):
            if not (FLAGS.special_first and is_first):
                net = self.BN_ReLU(net)

            # 1D-convolution
            net = self.conv2d(net, num_ker, kernel_size, stride)
        return net

    def conv_post(self, name, net, num_ker, kernel_size, stride, is_first):
        """ 1D post-activation convolution.
        args:
            num_ker (int): number of kernels (out_channels).
            ker_size (int): size of 1D kernel.
            stride (int)
        """
        with tf.variable_scope(name):
            # 1D-convolution
            net = self.conv2d(net, num_ker, kernel_size, stride)
            net = self.BN_ReLU(net)
        return net

    def residual_unit(self, net, group_i, unit_i):
        """pre-activation Residual Units from
        https://arxiv.org/abs/1603.05027."""
        name = 'group_%d/unit_%d' % (group_i, unit_i)
        group = self.groups[group_i]

        if group.is_downsample and unit_i == 0:
            stride1 = 2
        else:
            stride1 = 1

        ### residual function
        net_residual = net
        if FLAGS.unit_type == 0 and not FLAGS.special_first:
            unit_conv = self.conv_post
        elif FLAGS.unit_type == 1:
            unit_conv = self.conv_pre
        else:
            raise ValueError("wrong residual unit type:{}".format(
                FLAGS.unit_type))

        if FLAGS.residual_type == 0:
            is_first = True if group_i == unit_i == 0 else False
            # 1x1 convolution responsible for reducing dimension
            net_residual = unit_conv(name + '/conv_reduce', net_residual,
                                     group.num_ker, FLAGS.bott_size_ends,
                                     stride1, is_first)
            # 3x1 convolution bottleneck
            net_residual = unit_conv(name + '/conv_bottleneck', net_residual,
                                     group.num_ker, FLAGS.bott_size_mid, 1,
                                     False)
            # 1x1 convolution responsible for restoring dimension
            net_residual = unit_conv(name + '/conv_restore', net_residual,
                                     group.num_key_exp, FLAGS.bott_size_ends,
                                     1, False)
        elif FLAGS.residual_type == 1:
            net_residual = unit_conv(name + '/conv_one', net_residual,
                                     group.num_ker, FLAGS.bott_size_mid,
                                     stride1, False)
            # # if self.if_drop and group_i == 2:
            # if self.if_drop and unit_i == 0:
            #     with tf.name_scope("dropout"):
            #         net_residual = tf.nn.dropout(net_residual,
            #                                      self.dropout_keep_prob)
            net_residual = unit_conv(name + '/conv_two',
                                     net_residual,
                                     group.num_ker,
                                     FLAGS.bott_size_mid,
                                     1, )
        else:
            raise ValueError("residual_type error")

        ### shortcut connection
        num_ker_in = utils.last_dimension(net.get_shape(), min_rank=4)
        if FLAGS.shortcut == 0 and unit_i == 0:
            # average pooling for data downsampling
            if group.is_downsample:
                net = tf.nn.avg_pool(net,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
            # zero-padding for increasing kernel numbers
            if group.num_ker / num_ker_in == 2:
                net = tf.pad(net, [[0, 0], [0, 0], [0, 0],
                                   [int(num_ker_in / 2), int(num_ker_in / 2)]])
            elif group.num_ker != num_ker_in:
                raise ValueError("illigal kernel numbers at group {} unit {}"
                                 .format(group_i, unit_i))
        elif FLAGS.shortcut == 1 and unit_i == 0 or FLAGS.shortcut == 2:
            with tf.variable_scope(name + '_sc'):
                # projection
                net = self.BN_ReLU(net)
                net = self.conv2d(net, group.num_key_exp, 1, stride1)

        ### element-wise addition
        net = net + net_residual

        return net


    def resnn(self, image_batch):
        raise NotImplementedError("Should have implemented this")
        return

    def inference(self, feature_batch):
        """Build the resnn model.
        Args:
            feature_batch: for cifar-10:
                [batch_size, FLAGS.image_size, FLAGS.image_size, 3]
        Returns:
            Logits.
        """
        groups = FLAGS.groups_conf.split('\n')
        self.groups = []
        for group in groups:
            # remove the split after last \n
            if len(group) > 3:
                tmp = [int(i) for i in group.split(',')]
                tmp[1] = tmp[1] * FLAGS.wide_factor
                tmp[2] = tmp[2] * FLAGS.wide_factor
                self.groups.append(UnitsGroup(*tmp))

        return self.resnn(feature_batch)
