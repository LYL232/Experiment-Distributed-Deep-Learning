from tensorflow.python.eager import context
import tensorflow as tf


def executing_eagerly():
    return context.executing_eagerly()


def make_tf_function(f):
    return tf.function(f)
