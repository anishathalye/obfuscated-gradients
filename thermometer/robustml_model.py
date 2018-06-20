import robustml
import tensorflow as tf
from discretization_utils import discretize_uniform
import numpy as np
from cifar_model import Model

LEVELS = 16

class Thermometer(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess

        self._x = tf.placeholder(tf.float32, (1, 32, 32, 3))
        self._encode = discretize_uniform(self._x/255.0, levels=LEVELS, thermometer=True)

        self._model = Model(
            '../models/thermometer_advtrain/',
            sess,
            tiny=False,
            mode='eval',
            thermometer=True,
            levels=LEVELS
        )

        self._dataset = robustml.dataset.CIFAR10()
        self._threat_model = robustml.threat_model.Linf(epsilon=8.0/255.0)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        x = x * 255.0
        # first encode the input, then classify it
        encoded = self.encode(x)
        return self._sess.run(self._model.predictions, {self._model.x_input: encoded})[0]

    # expose internals for white box attacks

    @property
    def model(self):
        return self._model

    # x should be in [0, 255]
    def encode(self, x):
        return self._sess.run(self._encode, {self._x: [x]})
