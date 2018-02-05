import models.pixel_cnn_pp.nn as nn
from models.pixel_cnn_pp.model import model_spec
from utils import optimistic_restore
import tensorflow as tf
import numpy as np
import os

_PIXELCNN_CHECKPOINT_NAME = 'params_cifar.ckpt'
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'data'
)

_obs_shape = (32, 32, 3)

_model_opt = {
    'nr_resnet': 5,
    'nr_filters': 160,
    'nr_logistic_mix': 10,
    'resnet_nonlinearity': 'concat_elu'
}

# XXX this being called "model" could cause problems if other things want to use the same scope
_model_func = tf.make_template('model', model_spec)

def _init_model(sess, checkpoint_name=None):
    global _model_func
    global _obs_shape
    global _model_opt

    if checkpoint_name is None:
        checkpoint_name = _PIXELCNN_CHECKPOINT_NAME
    checkpoint_path = os.path.join(DATA_DIR, checkpoint_name)

    x_init = tf.placeholder(tf.float32, (1,) + _obs_shape)
    model = _model_func(x_init, init=True, dropout_p=0.5, **_model_opt)
    # XXX need to add a scope argument to optimistic_restore and filter for
    # things that start with "{scope}/", so we can filter for "model/", because
    # the pixelcnn checkpoint has some random unscoped stuff like 'Variable'
    optimistic_restore(sess, checkpoint_path)

# input is [batch, 32, 32, 3], pixels in [-1, 1]
_initialized=False
_initialized_name=None
def model(sess, image, checkpoint_name=None):
    global _initialized
    global _initialized_name
    global _model_func
    global _model_opt

    if checkpoint_name is not None:
        checkpoint_name = os.path.basename(checkpoint_name)
        # currently, we only support one version of this model loaded at a
        # time; making multiple versions probably involves variable renaming or
        # something else that's probably painful
        assert not _initialized or _initialized_name == checkpoint_name

    if not _initialized:
        _init_model(sess, checkpoint_name)
        _initialized = True
        _initialized_name = checkpoint_name

    out = _model_func(image, dropout_p=0, **_model_opt)
    loss = nn.discretized_mix_logistic_loss(image, out)
    return loss, out
