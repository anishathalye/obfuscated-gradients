import sys
import scipy.misc

import tensorflow as tf
import numpy as np
from cifar_model import SAPModel
from baseline.cifar_model import Model
import baseline.cifar10_input

import l2_attack

def verify(mod, data):
    xin = tf.placeholder(tf.float32, (batch_size, 32, 32, 3))
    logits = mod(xin)
    
    r = []
    for i in range(0,len(data),batch_size):
        s = []
        for _ in range(10):
            outs = sess.run(logits, {xin: data[i:i+batch_size]})
            s.append(np.argmax(outs,axis=1))
        same = np.min(s,axis=0)==np.max(s,axis=0)

        right = s[0] == cifar.eval_data.ys[i:i+batch_size]
        r.append(np.mean(right & same))
    print('accuracy',np.mean(r))

batch_size = 100

class SimplifiedCW:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        
        self.modifier = tf.Variable(np.zeros((100, 32, 32, 3),dtype=np.float32),name='modifier')

        self.xs = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.ys = tf.placeholder(tf.int32, [None])

        modifier = tf.clip_by_value(self.modifier, -epsilon, epsilon)
        self.logits = logits = model(tf.clip_by_value(self.xs+modifier, 0, 255))
        
        label_mask = tf.one_hot(self.ys,
                                10,
                                on_value=1.0,
                                off_value=0.0,
                                dtype=tf.float32)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)
                
        self.loss = (correct_logit - wrong_logit)
        
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.GradientDescentOptimizer(step_size*.1)
        #optimizer = tf.train.AdamOptimizer(step_size*1)

        grad,var = optimizer.compute_gradients(self.loss, [self.modifier])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        
        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)
        sess.run(tf.variables_initializer(self.new_vars))

        sess.run(self.modifier.initializer)
        for i in range(self.num_steps):
            loss,fakeres = sess.run((self.loss,self.logits), {self.xs: x, self.ys: y})
            if i%10 == 0:
                print(np.mean(loss),np.argmax(fakeres,axis=1)==y)
            
            sess.run(self.train, feed_dict={self.xs: x,
                                            self.ys: y})
            
        return x

with tf.Session() as sess:
    cifar = baseline.cifar10_input.CIFAR10Data("../baseline/cifar10_data")
    model = SAPModel("../baseline/models/tiny", sess, tiny=True)

    original_model = Model("../baseline/models/tiny", sess, tiny=True)
    original_model.image_size = 32
    original_model.num_labels = 10
    original_model.num_channels = 3
    original_model.predict = original_model.__call__

    #verify(original_model)
    #verify(model)

    attack = SimplifiedCW(model, 8, 1000, 1, False, 'cw')
    attack.perturb(cifar.eval_data.xs[:100],
                   cifar.eval_data.ys[:100],
                   sess)
    
    #exit(0)
    exit(0)
    
    rand_models = [SAPModel("../baseline/models/tiny", sess, tiny=True,
                            fix=False) for _ in range(1)]
    #rand_models = [original_model]

    #import matplotlib.pyplot as plt
    #plt.imshow(cifar.eval_data.xs[0])
    #plt.show()
    
    attack = l2_attack.CarliniL2(sess, rand_models, {}, {},
                                 binary_search_steps=1,
                                 initial_const=1000000, batch_size=1,
                                 max_iterations=300,
                                 learning_rate=1, boxmin=0, boxmax=255,
                                 targeted=False, abort_early=False,
                                 confidence=100)
    
    adv = attack.attack(cifar.eval_data.xs[:batch_size], [np.eye(10)[x] for x in cifar.eval_data.ys[:batch_size]])

    scipy.misc.imsave("/tmp/normal.png", cifar.eval_data.xs[:batch_size].reshape((batch_size*32,32,3)))
    scipy.misc.imsave("/tmp/adv.png", adv.reshape((batch_size*32,32,3)))

    #plt.imshow(adv[0])
    #plt.show()

    # normal: 29
    print('distortion', np.mean(np.sum((adv-cifar.eval_data.xs[:batch_size])**2,axis=(1,2,3))**.5))
    
    print('distortion', np.sum((adv-cifar.eval_data.xs[:batch_size])**2,axis=(1,2,3))**.5)
    
    xin = tf.placeholder(tf.float32, (1000, 32, 32, 3))
    logits = model(xin)

    for i in range(batch_size):
        print('success', np.mean(np.argmax(sess.run(logits, {xin: [adv[i]]*1000}),axis=1)==cifar.eval_data.ys[i]))

    exit(0)
              
