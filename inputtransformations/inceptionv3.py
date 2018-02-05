from utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

_INCEPTION_CHECKPOINT_NAME = 'inception_v3.ckpt'
INCEPTION_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    'data',
    _INCEPTION_CHECKPOINT_NAME
)

def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

# input is [batch, ?, ?, 3], pixels in [0, 1]
# it's rescaled to [batch, 299, 299, 3] and shifted to [-1, 1]
# output is [batch, 1000] (imagenet classes)
_inception_initialized = False
def model(sess, image):
    global _inception_initialized
    network_fn = _get_model(reuse=_inception_initialized)
    size = network_fn.default_image_size
    preprocessed = _preprocess(image, size, size)
    logits, _ = network_fn(preprocessed)
    logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _inception_initialized:
        optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        _inception_initialized = True

    return logits, predictions
