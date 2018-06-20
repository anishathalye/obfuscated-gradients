import robustml
from robustml_model import InputTransformations
import sys
import argparse
import tensorflow as tf
import numpy as np

class BPDA(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=1000, learning_rate=0.1, lam=1e-6, debug=False):
        self._sess = sess

        self._model = model
        self._input = model.input
        self._l2_input = tf.placeholder(tf.float32, self._input.shape) # using BPDA, so we want this to pass the original adversarial example
        self._original = tf.placeholder(tf.float32, self._input.shape)
        self._label = tf.placeholder(tf.int32, ())
        one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        ensemble_labels = tf.tile(one_hot, (model.logits.shape[0], 1))
        self._l2 = tf.sqrt(2*tf.nn.l2_loss(self._l2_input - self._original))
        self._xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=ensemble_labels))
        self._loss = lam * tf.maximum(self._l2 - epsilon, 0) + self._xent
        self._grad, = tf.gradients(self._loss, self._input)

        self._epsilon = epsilon
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        adv = np.copy(x)
        for i in range(self._max_steps):
            adv_def = self._model.defend(adv)
            p, ll2, lxent, g = self._sess.run(
                [self._model.predictions, self._l2, self._xent, self._grad],
                {self._input: adv_def, self._label: y, self._l2_input: adv, self._original: x}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, xent loss = %g, l2 loss = %g (max %g), (true %d, predicted %s)' % (
                        i+1,
                        self._max_steps,
                        lxent,
                        ll2,
                        self._epsilon,
                        y,
                        p
                    ),
                    file=sys.stderr
                )
            if y not in p and ll2 < self._epsilon:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += self._learning_rate * g
            adv = np.clip(adv, 0, 1)
        return adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--defense', type=str, required=True,
            help='bitdepth | jpeg | crop | quilt | tv')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set up TensorFlow session
    sess = tf.Session()

    # initialize a model
    model = InputTransformations(sess, args.defense)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # XXX restore
    attack = BPDA(sess, model, model.threat_model.epsilon, debug=args.debug)

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
