"""neural network model."""
import math

import tensorflow as tf
import numpy as np

import inputs
from utils import logger
from utils import FLAGS


class Model(object):
    """super class for neural network models."""

    def __init__(self, is_train=False):
        # self.num_cats = FLAGS.num_cats

        if is_train:
            # build training graph
            self.dropout_keep_prob = FLAGS.dropout_keep_prob

            self.global_step = tf.get_variable(
                "global_step",
                initializer=tf.zeros_initializer(
                    [],
                    dtype=tf.int64),
                trainable=False)

            # get input data
            feature_batch, label_batch = inputs.inputs(is_train=is_train)
            if FLAGS.debug:
                label_batch = tf.Print(label_batch,
                                       [label_batch],
                                       message='\nlabel_batch:',
                                       summarize=128)

            # Build a Graph that computes the logits predictions from the
            self.logits = self.inference(feature_batch)

            # Calculate predictions.
            self.top_k_op = tf.nn.in_top_k(self.logits, label_batch,
                                           FLAGS.top_k_train)
            tf.scalar_summary("accuracy",
                              tf.reduce_mean(tf.cast(self.top_k_op, "float")))

            # Calculate loss.
            self.loss = self.loss(self.logits, label_batch)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            self.train_op = self.training(self.loss, self.global_step)
        else:
            # build eval graph
            self.dropout_keep_prob = 1

            feature_batch_eval, label_batch_eval = inputs.inputs(
                is_train=False)
            self.logits_eval = self.inference(feature_batch_eval)
            # Calculate predictions.
            self.top_k_op_eval = tf.nn.in_top_k(
                self.logits_eval, label_batch_eval, FLAGS.top_k_eval)
            tf.scalar_summary(
                "accuracy_eval (batch)",
                tf.reduce_mean(tf.cast(self.top_k_op_eval, "float")))

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measure the sparsity of activations.
        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        # Error: these summaries cause high classifier error!!!
        # All inputs to node MergeSummary/MergeSummary must be from the same frame.

        # tensor_name = re.sub('%s_[0-9]*/' % "tower", '', x.op.name)
        # tf.histogram_summary(tensor_name + '/activations', x)
        # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def inference(self, feature_batch):
        raise NotImplementedError("Should have implemented this")
        return

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
        Returns:
            Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            labels,
            name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')
        # from tensorflow.python.ops import variables
        # added to the collection GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
        tf.add_to_collection('REGULARIZATION_LOSSES', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        total_loss = tf.add_n(
            tf.get_collection('REGULARIZATION_LOSSES'),
            name='total_loss')
        tf.scalar_summary(total_loss.op.name, total_loss)

        return total_loss

    def training(self, total_loss, global_step):
        """Train CNN model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
            train_op: op for training.
        """
        # Variables that affect learning rate.
        num_batches_per_epoch = FLAGS.num_train_examples / FLAGS.batch_size
        logger.info("num_batches_per_epoch: {}".format(num_batches_per_epoch))
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        logger.info("decay_steps: {}".format(decay_steps))

        # Decay the learning rate exponentially based on the number of steps.
        lr_decay = tf.train.exponential_decay(FLAGS.initial_lr,
                                              global_step,
                                              decay_steps,
                                              FLAGS.lr_decay_factor,
                                              staircase=True)
        # compare with 0.01 * 0.5^10
        lr = tf.maximum(lr_decay, 0.1**FLAGS.min_lr)
        tf.scalar_summary('learning_rate', lr)

        # optimizer = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        return train_op

    def train_step(self, sess):
        """run one step on one batch trainning examples."""
        step, _, loss_value, top_k = sess.run([self.global_step, self.train_op,
                                               self.loss, self.top_k_op])
        return step, loss_value, top_k

    def eval_once(self, sess):
        # it's better to divide exactly with no remainder
        num_iter = int(math.ceil(FLAGS.num_test_examples / FLAGS.batch_size))
        true_count = 0  # counts the number of correct predictions.
        total_sample_count = num_iter * FLAGS.batch_size
        eval_step = 0
        while eval_step < num_iter:
            predictions = sess.run([self.top_k_op_eval])
            true_count += np.sum(predictions)
            eval_step += 1

        # compute precision @ 1.
        precision = true_count / total_sample_count
        return precision
