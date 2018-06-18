import robustml
from robustml_model import PixelDefend
import sys
import argparse
import tensorflow as tf
import numpy as np

class BPDA(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=100, learning_rate=0.5, debug=False):
        self._sess = sess
        self._model = model
        self._grad, = tf.gradients(self._model.xent, self._model.x_input)

        self._epsilon = epsilon
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

        # self._input = tf.placeholder(tf.float32, (299, 299, 3))
        # x_expanded = tf.expand_dims(self._input, axis=0)
        # ensemble_xs = tf.concat([defend(x_expanded) for _ in range(sample_size)], axis=0)
        # self._logits, self._preds = inceptionv3_model(sess, ensemble_xs)

        # self._label = tf.placeholder(tf.int32, ())
        # one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        # ensemble_labels = tf.tile(one_hot, (self._logits.shape[0], 1))
        # self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=ensemble_labels))
        # self._grad, = tf.gradients(self._loss, self._input)

        # self._epsilon = epsilon
        # self._max_steps = max_steps
        # self._learning_rate = learning_rate
        # self._debug = debug

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        adv = np.copy(x)
        lower = np.clip(x - self._epsilon, 0, 1)
        upper = np.clip(x + self._epsilon, 0, 1)
        for i in range(self._max_steps):
            adv_purified = self._model.purify(adv)
            p, l, g = self._sess.run(
                [self._model.predictions, self._model.xent, self._grad],
                {self._model.x_input: [adv_purified], self._model.y_input: [y]}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %d)' % (i+1, self._max_steps, l, y, p),
                    file=sys.stderr
                )
            if y != p:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += self._learning_rate * np.sign(g[0])
            adv = np.clip(adv, lower, upper)
        return adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar-path', type=str, required=True,
            help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set up TensorFlow session
    sess = tf.Session()

    # initialize a model
    model = PixelDefend(sess)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)

    # initialize a data provider for CIFAR-10 images
    provider = robustml.provider.CIFAR10(args.cifar_path)

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=args.debug,
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
