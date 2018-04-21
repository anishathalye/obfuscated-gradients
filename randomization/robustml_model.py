import robustml
from defense import defend
from inceptionv3 import model as inceptionv3_model
import tensorflow as tf

class Randomization(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (299, 299, 3))
        input_expanded = tf.expand_dims(self._input, axis=0)
        randomized = defend(input_expanded)
        self._logits, self._predictions = inceptionv3_model(sess, randomized)
        self._dataset = robustml.dataset.ImageNet((299, 299, 3))
        self._threat_model = robustml.threat_model.Linf(epsilon=8.0/255.0)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        return self._sess.run(self._predictions, {self._input: x})[0]

    # expose internals for white box attacks

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
