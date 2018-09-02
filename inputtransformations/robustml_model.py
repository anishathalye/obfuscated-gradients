import robustml
from defense import *
from inceptionv3 import model as inceptionv3_model
import tensorflow as tf

class InputTransformations(robustml.model.Model):
    def __init__(self, sess, defense):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (299, 299, 3))
        input_expanded = tf.expand_dims(self._input, axis=0)

        if defense == 'crop':
            cropped_xs = defend_crop(self._input)
            self._logits, _ = inceptionv3_model(sess, cropped_xs)
            self._probs = tf.reduce_mean(tf.nn.softmax(self._logits), axis=0, keepdims=True)
        else:
            self._logits, _ = inceptionv3_model(sess, input_expanded)
            self._probs = tf.nn.softmax(self._logits)

        self._predictions = tf.argmax(self._probs, 1)

        if defense == 'bitdepth':
            self._defend = defend_reduce
        elif defense == 'jpeg':
            self._defend = defend_jpeg
        elif defense == 'crop':
            self._defend = lambda x: x # implemented as part of model so it's differentiable
        elif defense == 'quilt':
            self._defend = make_defend_quilt(sess)
        elif defense == 'tv':
            self._defend = defend_tv
        else:
            raise ValueError('invalid defense: %s' % defense)

        self._dataset = robustml.dataset.ImageNet((299, 299, 3))
        self._threat_model = robustml.threat_model.L2(epsilon=0.05*299) # 0.05 * sqrt(299*299)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        x_defended = self.defend(x)
        return self._sess.run(self._predictions, {self._input: x_defended})[0]

    # expose internals for white box attacks

    def defend(self, x):
        return self._defend(x)

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
