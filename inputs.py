"""This code process TFRecords text classification datasets.
YOU MUST run convertbefore running this (but you only need to
run it once).
"""
import os
import collections
import sys
from six.moves import urllib
import tarfile

import tensorflow as tf
# from tensorflow.models.image.cifar10 import cifar10
# from tensorflow.models.image.cifar10 import cifar10_input
# from tensorflow.models.image.cifar10.cifar10_input import _generate_image_and_label_batch
from tensorflow.models.image.cifar10.cifar10_input import read_cifar10

# import utils
from utils import logger
from utils import FLAGS

#########################################
# FLAGS
#########################################
FLAGS.add('--data_dir',
          type=str,
          default='data/',
          help='directory for storing datasets and outputs.')
FLAGS.add('--num_read_threads',
          type=int,
          default=5,
          help='number of reading threads to shuffle examples '
          'between files.')
FLAGS.add("--max_images",
          type=int,
          help="save up to max_images number of images in summary.")

# dataset specific settings
FLAGS.add('--dataset',
          type=str,
          default='cifar-10',
          help='dataset type, each dataset has its own default settings.')
FLAGS.add('--dataset_fld',
          type=str,
          help='overwrite the default tfr folder name under data_dir')
FLAGS.add("--num_cats",
          type=int,
          help="overwrite the nuber of categories of dataset.")
FLAGS.add("--num_train_examples",
          type=int,
          help="overwrite the number of training examples per epoch.")
FLAGS.add("--num_test_examples",
          type=int,
          help="overwrite the number of testing examples per epoch.")
FLAGS.add("--image_size",
          type=int,
          help="overwrite the default image size, squre image.")

# dataset default settings
DataConf = collections.namedtuple(
    'DataConf', ['num_train', 'num_test', 'num_cats', 'folder', 'image_size'])
DATA_CONF = {
    'mnist': DataConf(2000, 500, 5, '', 20),
    'cifar-10': DataConf(50000, 10000, 10, '', 24),
    'cifar-100': DataConf(5000, 10000, 100, '', 24),
}

# if following dataset flags are None, use default settings.
data_conf = DATA_CONF[FLAGS.get('dataset')]
FLAGS.overwrite_none(num_train_examples=data_conf.num_train,
                     num_test_examples=data_conf.num_test,
                     num_cats=data_conf.num_cats,
                     dataset_fld=data_conf.folder,
                     image_size=data_conf.image_size)

#########################################
# global variables
#########################################
# CATEGORIES = FLAGS.categories.split(',')
# Constants used for dealing with the files, matches convert_to_records.
TFR_SUFFIX = '.TFR'

#########################################
# functions
#########################################


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        logger.newline()
        statinfo = os.stat(filepath)
        logger.info('Successfully downloaded', filename, statinfo.st_size,
                    'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, smr_name):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary(smr_name, images, max_images=FLAGS.max_images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size, num_epochs):
    """Construct distorted input for CIFAR training using the Reader ops.
    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, FLAGS.image_size, FLAGS.image_size, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames, num_epochs)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = FLAGS.image_size
    width = FLAGS.image_size

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2,
                                               upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(FLAGS.num_train_examples *
                             min_fraction_of_examples_in_queue)
    logger.info('Filling queue with %d CIFAR images before starting to train. '
                'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True,
                                           smr_name='image_train')


def eval_inputs(data_dir, batch_size, num_epochs):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, FLAGS.image_size, FLAGS.image_size, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # if not eval_data:
    #     filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
    #                 for i in xrange(1, 6)]
    #     num_examples_per_epoch = FLAGS.num_train_examples
    # else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = FLAGS.num_test_examples

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames, num_epochs)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = FLAGS.image_size
    width = FLAGS.image_size

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           read_input.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=False,
                                           smr_name='image_eval')


def inputs(is_train=True):
    if FLAGS.dataset == 'cifar-10':
        # check downloaded
        maybe_download_and_extract()
        data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

        if is_train:
            images, labels = distorted_inputs(data_dir=data_dir,
                                              batch_size=FLAGS.batch_size,
                                              num_epochs=FLAGS.num_epochs)
        else:
            images, labels = eval_inputs(data_dir=data_dir,
                                         batch_size=FLAGS.batch_size,
                                         num_epochs=None)

    return images, labels
