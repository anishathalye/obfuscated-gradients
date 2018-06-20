import robustml
import tensorflow as tf
import numpy as np
from sap_model import SAPModel

class SAP(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess

        self._model = SAPModel(
            '../models/standard/',
            tiny=False,
            mode='eval',
            sess=sess,
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
        return self._sess.run(self._model.predictions, {self._model.x_input: [x]})[0]

    # expose internals for white box attacks

    @property
    def model(self):
        return self._model
