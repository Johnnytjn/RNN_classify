import codecs
import csv
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def single_rnn_cell(cell_name, num_units, train_phase=True, keep_prob=0.75, weight_keep_drop=0.65, variational_dropout = False):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(num_units)
    elif cell_name == 'WEIGHT_LSTM':
        cell = WeightDropLSTMCell(num_units,weight_keep_drop=weight_keep_drop,mode=tf.estimator.ModeKeys.TRAIN if train_phase else tf.estimator.ModeKeys.PREDICT)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)

    # dropout wrapper
    if train_phase and keep_prob < 1.0:
        # TODO: variational_recurrent=True and input_keep_prob < 1 then we need provide input_size
        # But because we use different size in different layers, we will got shape in-compatible error
        # So I just set input_keep_prob to 1.0 when we use variational dropout to avoid this error for now.
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob if not variational_dropout else 1.0,
            output_keep_prob=keep_prob,
            variational_recurrent=variational_dropout,
            dtype=tf.float32)

    return cell

def residual_rnn_cell(residual_type):
    if residual_type.lower() == 'residual':
        return tf.contrib.rnn.ResidualWrapper
    elif residual_type.lower() == 'highway':
        return tf.contrib.rnn.HighwayWrapper
    else:
        raise ValueError("%s residual type not supported!!"%residual_type)
