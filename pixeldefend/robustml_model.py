import robustml
import tensorflow as tf
import numpy as np
from model import Model
import models.pixelcnn_cifar as pixelcnn
from utils import *
from defense import *

class PixelDefend(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        self._model = Model(mode='eval')
        self._grad, = tf.gradients(self._model.xent, self._model.x_input)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('data/models/naturally_trained'))
        self._input = tf.placeholder(tf.float32, (1, 32, 32, 3))
        _, pixelcnn_out = pixelcnn.model(sess, self._input)
        self._pixeldefend = make_pixeldefend(sess, self._input, pixelcnn_out)
        self._dataset = robustml.dataset.CIFAR10()
        self._threat_model = robustml.threat_model.Linf(epsilon=8.0/255.0)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        # first purify the input, then classify it
        x_purified = self._pixeldefend(x)
        return self._sess.run(self._model.predictions, {self._model.x_input: [x_purified]})[0]

    # expose internals for white box attacks

    def purify(self, x):
        return self._pixeldefend(x)

    @property
    def x_input(self):
        return self._model.x_input

    @property
    def y_input(self):
        return self._model.y_input

    @property
    def predictions(self):
        return self._model.predictions

    @property
    def xent(self):
        return self._model.xent
