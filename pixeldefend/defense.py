import tensorflow as tf
import numpy as np
from utils import int_shape

def make_pixeldefend(sess, x, pixelcnn_out):
    l = pixelcnn_out
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x_ = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x_[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x_[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x_[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)

    # B is batch size (1)
    # H and W are height and width (32)
    # C is channels (3)
    # M is number of mixtures (10)
    # shapes are (B, H, W, C, M)

    eval_pts = tf.constant(np.linspace(-1+1./256, 1-1./256, 255, dtype=np.float32))
    eval_pts = tf.reshape(eval_pts, (1, 1, 1, 1, 1, -1))
    eval_pts = tf.tile(eval_pts, (xs[0],xs[1],xs[2],3,nr_mix,1))

    log_scales = tf.reshape(log_scales, (xs[0],xs[1],xs[2],3,nr_mix,1))
    scales = tf.exp(log_scales)
    scales = tf.tile(scales, (1, 1, 1, 1, 1, 255))

    means = tf.reshape(means, (xs[0],xs[1],xs[2],3,nr_mix,1))
    means = tf.tile(means, (1, 1, 1, 1, 1, 255))

    evals = tf.sigmoid((eval_pts - means) / scales)

    eval_upper = tf.concat([evals, tf.ones((xs[0],xs[1],xs[2],3,nr_mix,1))], axis=5)
    eval_lower = tf.concat([tf.zeros((xs[0],xs[1],xs[2],3,nr_mix,1)), evals], axis=5)

    eval_diffs = eval_upper - eval_lower

    probs_tiled = tf.nn.softmax(
        tf.tile(tf.reshape(logit_probs, (xs[0],xs[1],xs[2],1,nr_mix,1)), (1,1,1,3,1,256)),
        axis=4
    )

    probs = tf.reduce_sum(eval_diffs * probs_tiled, axis=4)

    # input image has elements in [0, 255]
    # epsilon is 0-255
    def pixeldefend(input_image, eps=16):
        purified = 2.0*np.copy(input_image)/255.0 - 1.0 # rescale to [-1, 1]
        
        for yi in range(32):
            for xi in range(32):
                # we have to do this one channel at a time, due to channel-wise dependencies
                for ki in range(3):
                    p = sess.run(probs, {x: [purified]})
                    sub = p[0,yi,xi,ki]
                    
                    curr_val = np.floor(255.0*(purified[yi,xi,ki]+1)/2.0)
                    feasible = range(int(max(curr_val-eps, 0)), int(min(curr_val+eps, 255)+1))
                    
                    best_p = -1
                    best_idx = None
                    for i in feasible:
                        if sub[i] > best_p:
                            best_p = sub[i]
                            best_idx = i
                    
                    purified[yi,xi,ki] = 2.0*best_idx/255.0 - 1.0
                    
        return 255.0*((purified+1.0)/2.0)

    return pixeldefend
