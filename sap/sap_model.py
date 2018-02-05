import tensorflow as tf
import numpy as np

from cifar_model import Model

class SAPModel(Model):
    image_size = 32
    num_labels = 10
    num_channels = 3

    def __init__(self,  *args, **kwargs):
        if 'fix' in kwargs:
            self.fix_randomness = kwargs['fix'] == True
            del kwargs['fix']
        else:
            self.fix_randomness = False
        super().__init__(*args, **kwargs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        r = super()._conv(name, x, filter_size, in_filters, out_filters, strides)
        r = tf.check_numerics(r, "okay")
        p = tf.abs(r)/tf.reduce_sum(tf.abs(r), axis=(1,2,3), keep_dims=True)
        w,h,c = p.get_shape().as_list()[1:]
        N = w*h*c*2
        if self.fix_randomness:
            p_keep = 1-tf.exp(-N*p)
            rand = tf.constant(np.random.uniform(size=(p_keep.shape[0],w,h,c)),
                               dtype=tf.float32)
        else:
            p_keep = 1-tf.exp(-N*p)
            rand = tf.random_uniform(tf.shape(p_keep))
        keep = rand<p_keep
        r = tf.cast(keep, tf.float32)*r/(p_keep+1e-8)
        r = tf.check_numerics(r, "OH NO")
        return r

    def _build_model(self, x_input = None):
        if x_input == None:
            x_input = tf.placeholder(tf.float32, (None, 32, 32, 3))
        return super()._build_model(x_input)

    def predict(self, x):
        return self.__call__(x)
