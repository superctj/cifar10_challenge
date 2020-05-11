"""
Wide-Resnet-34 Classifier for 32x32x3 CIFAR-10 Image Classification.
Reference: https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/cifar.py
"""
import os

import numpy as np
import tensorflow as tf
from armory import paths
from art.classifiers import TensorFlowClassifier

from arch.model import make_madry_model


def preprocessing_fn(img):
    # img = img.astype(np.float32) / 255.0
    # img = img.transpose(0, 3, 1, 2)  # from NHWC to NCHW
    img = img.astype(np.float32)
    # print(img.shape)

    return img


def get_madry_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_madry_model(**model_kwargs)
    saver = tf.train.Saver()
    tf_sess = tf.Session()

    # Restore the checkpoint
    saved_model_dir = paths.DockerPaths().saved_model_dir
    filepath = os.path.join(saved_model_dir, weights_file)
    model_file = tf.train.latest_checkpoint(filepath)
    saver.restore(tf_sess, model_file)

    wrapped_model = TensorFlowClassifier(
        input_ph=tf.placeholder(tf.float32, shape=[None, 32, 32, 3]),
        output=model.pre_softmax,
        labels_ph=tf.placeholder(tf.int64, shape=[None, 10]),
        loss=model.xent,
        sess=tf_sess,
        clip_values=(0, 255),
        **wrapper_kwargs
    )

    return wrapped_model