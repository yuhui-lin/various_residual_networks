import os
import time

import numpy as np
import tensorflow as tf

# from models import model
# import utils
from utils import CUR_TIME, logger, FLAGS

#########################################
# FLAGS
#########################################
# environmental parameters
FLAGS.add('--model',
          type=str,
          default='resnet',
          help='the type of NN model. cnn, crnn, resnn, resrnn...')
FLAGS.add('--outputs_dir',
          type=str,
          default='outputs/',
          help='Directory where to write event logs and checkpoint.')
FLAGS.add("--log_level",
          type=int,
          default=20,
          help="numeric value of logger level, 20 for info, 10 for debug.")
# FLAGS.add('--if_eval',
#           type=bool,
#           default=True,
#           help="Whether to log device placement.")
FLAGS.add('--debug',
          type=bool,
          default=False,
          help="whether to print debug infor")
FLAGS.add("--debug_len",
          type=int,
          default=200,
          help="length of debug information.")

# Training parameters
FLAGS.add("--num_epochs",
          type=int,
          default=200,
          help="Number of training epochs (default: 100)")
FLAGS.add("--batch_size",
          type=int,
          default=64,
          help="mini Batch Size (default: 64)")
FLAGS.add("--top_k_train",
          type=int,
          default=1,
          help="compare the top n results when training.")
FLAGS.add("--top_k_eval",
          type=int,
          default=1,
          help="compare the top n results when eval.")
FLAGS.add("--dropout_keep_prob",
          type=float,
          default=0.5,
          help="Dropout keep probability (default: 0.5)")
FLAGS.add("--momentum",
          type=float,
          default=0.9,
          help="the momentum index for SGD training.")
FLAGS.add('--weight_decay',
          type=float,
          default=0.0001,
          help='weight decay(l2 norm)')
# FLAGS.add('--max_steps',
#           type=int,
#           default=1000000,
#           help="""Max number of total batches to run.""")

# learning rate decay
FLAGS.add('--num_epochs_per_decay',
          type=int,
          default=30,
          help="number of epochs for every learning rate decay.")
FLAGS.add("--lr_decay_factor",
          type=float,
          default=0.1,
          help="learning rate decay factor.")
FLAGS.add("--initial_lr",
          type=float,
          default=0.1,
          help="inital learning rate.")
FLAGS.add('--min_lr', type=int, default=5, help="e^-n, minimum learning rate.")

# Misc Parameters
FLAGS.add("--allow_soft_placement",
          type=bool,
          default=True,
          help="Allow device soft device placement")
FLAGS.add('--log_device_placement',
          type=bool,
          default=False,
          help="""Whether to log device placement.""")
FLAGS.add('--print_step',
          type=int,
          default=1,
          help="""Number of steps to print current state.""")
FLAGS.add('--summary_step',
          type=int,
          default=3,
          help="""Number of steps to write summaries.""")
FLAGS.add('--eval_step',
          type=int,
          default=-1,
          help="""Number of steps to eval on test set. """)
FLAGS.add('--checkpoint_step',
          type=int,
          default=-1,
          help="""Number of steps to write checkpoint, -1 to disable.""")
FLAGS.add('--num_checkpoints',
          type=int,
          default=5,
          help="Number of maximum checkpoints to keep. default: 10")
FLAGS.add('--sleep',
          type=int,
          default=0,
          help="the number of seconds to sleep between steps. 0, 1, 2...")

#########################################
# global variables
#########################################
RESULT_FLD = FLAGS.get('model') + '_' + CUR_TIME
RESULT_DIR = os.path.join(FLAGS.get('outputs_dir'), RESULT_FLD)
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model.ckpt')

#########################################
# functions
#########################################


def train(model_class):
    """Train neural network for a number of steps."""
    logger.info("\nstart training...")
    with tf.Graph().as_default():
        # build computing graph
        with tf.variable_scope("model", reuse=None):
            model_train = model_class(is_train=True)
        if FLAGS.eval_step > 0:
            with tf.variable_scope("model", reuse=True):
                model_eval = model_class(is_train=False)

        saver = tf.train.Saver(tf.all_variables(),
                               max_to_keep=FLAGS.num_checkpoints)
        sv = tf.train.Supervisor(logdir=RESULT_DIR,
                                 saver=saver,
                                 save_summaries_secs=0,
                                 save_model_secs=0)

        logger.newline()
        logger.info("start building Graph (This might take a while)")
        # Start running operations on the Graph.
        sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement))

        logger.newline()
        logger.info("start training...")
        try:
            while not sv.should_stop():
                start_time = time.time()
                step, loss_value, top_k = model_train.train_step(sess)
                duration = time.time() - start_time

                assert not np.isnan(
                    loss_value), 'Model diverged with loss = NaN'

                # print current state
                if step % FLAGS.print_step == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    precision = np.sum(top_k) / FLAGS.batch_size

                    format_str = (
                        'step %d, loss = %.2f, precision = %.2f (%.1f '
                        'examples/sec; %.3f sec/batch)')
                    logger.info(format_str % (step, loss_value, precision,
                                              examples_per_sec, sec_per_batch))

                # save summary
                if step % FLAGS.summary_step == 0:
                    summary_str = sess.run(sv.summary_op)
                    sv.summary_writer.add_summary(summary_str, step)
                    logger.info("step: {}, wrote summaries.".format(step))

                # Save the model checkpoint periodically and eval on test set.
                if FLAGS.checkpoint_step > 0 and step % FLAGS.checkpoint_step == 0:
                    saver_path = sv.saver.save(sess,
                                               CHECKPOINT_PATH,
                                               global_step=step)
                    logger.newline(2)
                    logger.info("Saved model checkpoint to {}\n\n".format(
                        saver_path))

                if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                    logger.newline(2)
                    logger.info("evaluating current model:")
                    precision = model_eval.eval_once(sess)
                    logger.info('%s: precision @ 1 = %.3f' %
                                (time.strftime("%c"), precision))

                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(sv.summary_op))
                    summary.value.add(tag='precision @ 1',
                                      simple_value=precision)
                    sv.summary_writer.add_summary(summary, step)
                    logger.info("write eval summary\n\n")

                # sleep for test use
                if FLAGS.sleep > 0:
                    logger.info("sleep {} second...".format(FLAGS.sleep))
                    time.sleep(FLAGS.sleep)
        except tf.errors.OutOfRangeError:
            logger.info("sv checkpoint saved path: " + sv.save_path)
            logger.info("Done~\n\n")
        finally:
            sv.request_stop()
        sv.wait_for_stop()
        sess.close()


def main(argv=None):
    print("start of main")
    main_time = time.time()

    os.makedirs(RESULT_DIR)

    # loging
    LOG_FILE = os.path.join(RESULT_DIR, "log.txt")
    logger.set_logger(level=FLAGS.get('log_level'),
                      stream=True,
                      fileh=True,
                      filename=LOG_FILE)

    # file handling
    logger.info("create folder for results: {}".format(RESULT_DIR))
    if FLAGS.checkpoint_step > 0:
        os.mkdir(CHECKPOINT_DIR)
        logger.info("create checkpoints folder: {}".format(CHECKPOINT_DIR))

    # import the corresponding module
    # what about models.model ????????
    try:
        model_path = 'models.' + FLAGS.get('model').lower()
        model_module = __import__(model_path, fromlist=[''])
    except ImportError:
        raise ValueError("no such model exists: {}".format())

    # parse all FLAGS
    FLAGS.parse_and_log()

    # start training
    train(model_module.Model)

    # pring something before end
    logger.newline(2)
    logger.info("total time used: " + time.time() - main_time)
    logger.info("summary dir: " + RESULT_DIR)
    logger.newline()
    logger.info("~end of main~")


if __name__ == '__main__':
    tf.app.run()
