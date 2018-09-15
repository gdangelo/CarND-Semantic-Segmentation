#!/usr/bin/env python3
import sys
import os
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import time
import math
import argparse
from glob import glob
import project_tests as tests

FLAGS = None

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer. You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load pretrained VGG model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Get default graph
    graph = tf.get_default_graph()

    # Retrieve tensors from graph
    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input, keep_prob, layer3_out, layer4_out, layer7_out

#tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, l2_scale, normal_stddev):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    l2_reg = tf.contrib.layers.l2_regularizer(l2_scale)
    normal_init = tf.random_normal_initializer(stddev=normal_stddev)

    # Add 1x1 convolution
    layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                    kernel_initializer=normal_init,
                                    kernel_regularizer=l2_reg)
    layer4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                    kernel_initializer=normal_init,
                                    kernel_regularizer=l2_reg)
    layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                    kernel_initializer=normal_init,
                                    kernel_regularizer=l2_reg)

    # Upsample 7th using backward stride convolutions (x2)
    layer7_conv1x1_2x = tf.layers.conv2d_transpose(layer7_conv1x1, num_classes,
                                                kernel_size=4,
                                                strides=(2, 2),
                                                padding='same',
                                                kernel_initializer=normal_init,
                                                kernel_regularizer=l2_reg)

    # Add first skip layer
    skip1 = tf.add(layer4_conv1x1, layer7_conv1x1_2x)

    # Upsample first skip layer using backward stride convolutions (x2)
    skip1_2x = tf.layers.conv2d_transpose(skip1, num_classes,
                                        kernel_size=4,
                                        strides=(2, 2),
                                        padding='same',
                                        kernel_initializer=normal_init,
                                        kernel_regularizer=l2_reg)

    # Add second skip layer
    skip2 = tf.add(layer3_conv1x1, skip1_2x)

    # Upsample to output segmentation
    output = tf.layers.conv2d_transpose(skip2, num_classes,
                                        kernel_size=16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_initializer=normal_init,
                                        kernel_regularizer=l2_reg)

    return output

#tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape output tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Build TensorFlow cross entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Get L2 Loss
    l2_loss = tf.losses.get_regularization_loss()

    # Compute total loss
    total_loss = cross_entropy_loss + l2_loss

    # Compute accuracy and IOU metrics
    predictions = tf.argmax(tf.nn.softmax(logits), -1)
    labels = tf.argmax(correct_label, -1)
    accuracy, accuracy_update = tf.metrics.accuracy(labels, predictions)
    mean_iou, mean_iou_update = tf.metrics.mean_iou(labels, predictions, num_classes)
    metrics_op = tf.group(accuracy_update, mean_iou_update)

    # Build TensorFlow Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    return logits, train_op, total_loss, accuracy, mean_iou, metrics_op

#tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, keep_prob_value, get_batches_fn, train_op,
            cross_entropy_loss, accuracy, mean_iou, metrics_op,
            input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param keep_prob_value: Dropout keep probability value
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param accuracy: TF Tensor representing the accuracy
    :param mean_iou: TF Tensor representing the mean intersection-over-union
    :param metrics_op: TF Tensor representing group operator for metrics (accuracy, iou)
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print('\nStarting training for {} epochs\n'.format(epochs))

    # Init variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Go through each epoch
    for i in range(epochs):
        print("EPOCH {}".format(i+1))

        # Generate bacthes of data
        for images, gt_images in get_batches_fn(batch_size):
            # Run training
            t0 = time.time()
            _, loss, acc, iou, lr, _ = sess.run([train_op, cross_entropy_loss, accuracy, mean_iou, learning_rate, metrics_op], feed_dict={input_image: images, correct_label: gt_images, keep_prob: keep_prob_value})
            t1 = time.time()
            time_spent = int(round((t1-t0)*1000))

            # After each run, print metrics
            print("Loss = {:.3f}, Accuracy = {:.4f}, IOU = {:.4f}, LR = {:.2e} | time = {}ms".format(loss, acc, iou, lr, time_spent))

        print()

#tests.test_train_nn(train_nn)

def learning_rate(global_step, initial_learning_rate, decay_rate, num_epochs_per_decay, batch_size):
    """
    Define exponentially decaying learning rate.
    :param global_step: Variable counting the number of training steps processed
    :param initial_learning_rate: Initial learning rate
    :param decay_rate: Rate factor for learning rate
    :param num_epochs_per_decay: Number of epochs after which we should decay the learning rate
    :param batch_size: Batch size
    :return: Learning rate
    """

    # Compute number of batches per epoch
    nb_images = len(glob(os.path.join('./data/data_road/training', 'image_2', '*.png')))
    num_batches_per_epoch = math.ceil(nb_images / float(batch_size))

    # Compute decay steps for learning rate
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    print(decay_steps)

    # Define learning rate
    return tf.train.exponential_decay(
                learning_rate = initial_learning_rate,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = decay_rate,
                staircase = True)

def run(_):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize functions
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, FLAGS.L2_SCALE, FLAGS.STDDEV)

        # Define exp dacay learning rate
        global_step = tf.train.get_or_create_global_step()
        lr = learning_rate(global_step, FLAGS.INIT_LEARNING_RATE, FLAGS.LR_DECAY_FACTOR, FLAGS.EPOCHS_PER_DECAY, FLAGS.BATCH_SIZE)

        # Build the TF loss, metrics, and optimizer operations
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        logits, train_op, cross_entropy_loss, accuracy, mean_iou, metrics_op = optimize(nn_last_layer, correct_label, lr, num_classes, global_step)

        # Train NN using the train_nn function
        train_nn(sess, FLAGS.EPOCHS, FLAGS.BATCH_SIZE, FLAGS.KEEP_PROB, get_batches_fn, train_op, cross_entropy_loss, accuracy, mean_iou, metrics_op, input_image, correct_label, keep_prob, lr)

        # Save the all the graph variables to disk
        saver = tf.train.Saver()
        if not os.path.isdir("./model"):
            os.makedirs("./model")
        save_path = saver.save(sess, "./model/model.ckpt")
        print("Model saved in path: %s \n" % save_path)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # Apply the trained model to a video
        video_path = './videos/test_video.mp4'
        #helper.inference_on_video(video_path, image_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--EPOCHS', default=40, type=int, help='Number of epochs')
    parser.add_argument('--BATCH_SIZE', default=16, type=int, help='Batch size')
    parser.add_argument('--KEEP_PROB', default=0.5, type=float, help='Keep probabilty')
    parser.add_argument('--INIT_LEARNING_RATE', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--LR_DECAY_FACTOR', default=1e-1, type=float, help='Learning rate decay factor')
    parser.add_argument('--EPOCHS_PER_DECAY', default=20, type=int, help='Number of epochs per decay')
    parser.add_argument('--L2_SCALE', default=1e-3, type=float, help='Scale for L2 regularizers')
    parser.add_argument('--STDDEV', default=0.01, type=float, help='Standard deviation for random normal initializers')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
