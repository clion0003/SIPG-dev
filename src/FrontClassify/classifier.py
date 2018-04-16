from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
#import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
import random
import resnet_model

_HEIGHT = 224
_WIDTH = 224
_DEPTH = 3
_NUM_CLASSES = 2

_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_TRAIN_TEST_RATE = 0.7
_DATASIZE = 0
_TRAIN_DATASIZE = 0
_TEST_DATASIZE = 0

_SHUFFLE_BUFFER = 1024

_BATCH_SIZE = 32
_EPOCH = 1000
_RESNET_SIZE = 50
_EPOCH_PER_EVAL = 50

def _parse_function(filename, label=None):
    '''read image from filename and return tf.image if predict or tf.image, one_hot label if train.'''
    image_string = tf.read_file(filename)
    # Decode image
    image_decoded = tf.image.decode_jpeg(image_string, channels = _DEPTH)
    # Resize image
    image_resized = tf.image.resize_images(image_decoded, [_HEIGHT, _WIDTH])
    # Convert image datatype to tf.float
    image = tf.image.convert_image_dtype(image_resized, dtype=tf.float32)
    if label is None:
        # predict
        return image
    else:
        # train
        return image, tf.one_hot(label, _NUM_CLASSES)

def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

def generate_train_test_data_set(data_dir=None):
    '''generate train test dataset, return lists of image filepath and image labels'''
    if data_dir is None:
        # Current path as default
        data_dir = os.getcwd()
    
    # Positive image path
    ppath = os.path.join(data_dir, 'postive')
    # Negative image path
    npath = os.path.join(data_dir, 'negative')
    
    # generate positive data list
    pdata = [[os.path.join(ppath, f), 1] for f in os.listdir(ppath)]
    # generate negative data list
    ndata = [[os.path.join(npath, f), 0] for f in os.listdir(npath)]
    
    # concatenate positive data and negative data
    data = np.concatenate([pdata, ndata], 0)
    # shuffle data
    np.random.shuffle(data)

    # calculate train_size and test_size
    data_size = int(len(data))

    train_size = int(data_size * _TRAIN_TEST_RATE)
    test_size = data_size - train_size

    #  set size info to global vals
    global _DATASIZE
    _DATASIZE = data_size
    global _TRAIN_DATASIZE
    _TRAIN_DATASIZE = train_size
    global _TEST_DATASIZE
    _TEST_DATASIZE = test_size

    # split data and label
    fname = data[:, 0]
    label = data[:, 1]
    label = label.astype(np.int32)

    # split train and test data set
    train_data, test_data = np.split(fname, [train_size])
    train_label, test_label = np.split(label, [train_size])
    
    return train_data, train_label, test_data, test_label

def input_fn(image, label, is_training, batch_size, num_epochs=_EPOCH_PER_EVAL):
    '''Use tf.dataset to generate train dataset with label'''
    tf_image = tf.constant(image)
    tf_label = tf.constant(label)
    
    # generate tf.dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf_image, tf_label))

    # parse filepath to image data
    dataset = dataset.map(_parse_function)
    # preprocess image data
    dataset = dataset.map(lambda image, label: (preprocess_image(image, is_training), label))
    
    # shuffle dataset
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
    
    # set epoch and batchsize
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    # generate dataset iterator
    iterator = dataset.make_one_shot_iterator()
    
    # generate a batch of data and label by iterator
    next_image, next_label = iterator.get_next()
    return next_image, next_label

def predict_input_fn(image):
    '''Generate predict dataset, no label'''
    tf_image = tf.constant(image)
    
    dataset = tf.data.Dataset.from_tensor_slices(tf_image)

    dataset = dataset.map(lambda image: _parse_function(image, None))
    dataset = dataset.map(lambda image: preprocess_image(image, False))
    
    dataset = dataset.repeat(1)
    dataset = dataset.batch(len(image))
    
    iterator = dataset.make_one_shot_iterator()
    
    next_image = iterator.get_next()
    return next_image
    

def resnet_model_fn(features, labels, mode, params):
    '''Resnet model interface'''
    tf.summary.image('images', features, max_outputs=6)
    
    # call offical resnet model
    network = resnet_model.imagenet_resnet_v2(
      params['resnet_size'], _NUM_CLASSES, params['data_format'])

    logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)
    
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                     if 'batch_normalization' not in v.name])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.
        initial_learning_rate = 0.1
        #print(initial_learning_rate)
        batches_per_epoch = _TRAIN_DATASIZE / params['batch_size']
        #print(_TRAIN_DATASIZE)
        #print( params['batch_size'])
        #print(batches_per_epoch)
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [200, 500, 800]]
        print(boundaries)
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        print(values)
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None
        
    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes.
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    # print(initial_learning_rate)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def predict(img_path, show_img=False):
    '''Predict interface'''
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # Reset classifier 
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, 
        model_dir='../../model_para/FrontClassify/resnet_model', 
        config=tf.estimator.RunConfig(save_checkpoints_secs=1e9, session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))),
        params={
          'resnet_size': _RESNET_SIZE,
          'data_format': None,
          'batch_size': _BATCH_SIZE,
        })

    if type(img_path)==list:
        # img_path is a list of image file path, predict them one pass
        predictions = list(resnet_classifier.predict(input_fn=lambda: predict_input_fn(img_path)))
        result = [p['classes'] for p in predictions]
    else:
        # img_path is one image file path, predict directly
        predictions = list(resnet_classifier.predict(input_fn=lambda: predict_input_fn([img_path])))
        result = predictions[0]['classes']
    
    print('Preciction result: ')
    print(predictions)
    
    # result 1 means input image has railings and is the back of the container box.
    # result 0 means no railings and is the front.
    return result
        
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    cwd = os.getcwd()
    test_path = os.path.join(cwd, 'test_img')
    test_img = [os.path.join(test_path, f) for f in os.listdir(test_path)]
    print('Starting to predict....')
    print('Input image file path: ')
    print(test_img)
    result = predict(test_img)
    print('Final Result: (1 presents the back and 0 presents the front.)')
    print(result)

