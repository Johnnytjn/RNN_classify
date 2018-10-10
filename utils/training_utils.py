import codecs
import csv
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def load_hparams(out_dir, overidded = None):
    hparams_file = os.path.join(out_dir,"hparams")
    print("loading hparams from %s" % hparams_file)
    hparams_json = json.load(open(hparams_file))
    hparams = tf.contrib.training.HParams()
    for k,v in hparams_json.items():
        hparams.add_hparam(k,v)
    if overidded:
        for k,v in overidded.items():
            if k not in hparams_json:
                hparams.add_hparam(k,v)
            else:
                hparams.set_hparam(k,v)
    return hparams

def save_hparams(out_dir, hparams):
    """Save hparams."""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    hparams_file = os.path.join(out_dir, "hparams")
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())

def get_config_proto(log_device_placement=True, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0, per_process_gpu_memory_fraction=0.95, allow_growth=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto

def early_stop(values, no_decrease=3):
    if len(values) < 2:
        return False
    best_index = np.argmin(values)
    if values[-1] > values[best_index] and (best_index + no_decrease) <= len(values):
        return True
    else:
        return False