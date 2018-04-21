import robustml
from defense import defend
from inceptionv3 import model as inceptionv3_model
from robustml_model import Randomization
import sys
import argparse
import tensorflow as tf
import numpy as np

class EOT(robustml.attack.Attack):
    def __init__(self, sess, epsilon, sample_size=30, max_steps=1000, learning_rate=0.1, debug=False):
        self._sess = sess

        self._input = tf.placeholder(tf.float32, (299, 299, 3))
        x_expanded = tf.expand_dims(self._input, axis=0)
        ensemble_xs = tf.concat([defend(x_expanded) for _ in range(sample_size)], axis=0)
        self._logits, self._preds = inceptionv3_model(sess, ensemble_xs)

        self._label = tf.placeholder(tf.int32, ())
        one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        ensemble_labels = tf.tile(one_hot, (self._logits.shape[0], 1))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits, labels=ensemble_labels))
        self._grad, = tf.gradients(self._loss, self._input)

        self._epsilon = epsilon
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        adv = np.copy(x)
        lower = np.clip(x - self._epsilon, 0, 1)
        upper = np.clip(x + self._epsilon, 0, 1)
        for i in range(self._max_steps):
            p, l, g = self._sess.run(
                [self._preds, self._loss, self._grad],
                {self._input: adv, self._label: y}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %s)' % (i+1, self._max_steps, l, y, p),
                    file=sys.stderr
                )
            if y not in p:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += self._learning_rate * g
            adv = np.clip(adv, lower, upper)
        return adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set up TensorFlow session
    sess = tf.Session()

    # initialize a model
    model = Randomization(sess)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    attack = EOT(sess, model.threat_model.epsilon, debug=args.debug)

    # initialize a data provider for ImageNet images
    provider = robustml.provider.ImageNet(args.imagenet_path, model.dataset.shape)

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
