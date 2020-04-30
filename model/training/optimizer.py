# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:37:16 2019

@author: brown
"""

import tensorflow as tf
import torch.optim as optim
from .radam import RAdam


def get_optimizer(cost, global_step, batch_steps_per_epoch, kwargs={}):
    optimizer_name = kwargs.get("optimizer", None)
    learning_rate = kwargs.get("learning_rate", None)
    decay_rate = kwargs.get("decay_rate", 0.985)
    decay_epochs = kwargs.get("decay_epochs", 1)
    decay_steps = decay_epochs * batch_steps_per_epoch
    with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
        if(optimizer_name is "momentum"):
            momentum = kwargs.get("momentum",0.9)
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                       global_step=global_step,
                                                       decay_rate=decay_rate,
                                                       decay_steps=decay_steps,
                                                       staircase=True,
                                                       )
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node,
                                                   momentum=momentum,
                                                   )
        elif(optimizer_name is "rmsprop"):
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_node)
        elif(optimizer_name is "adabound"):
            from .optimization.adabound import AdaBoundOptimizer
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = AdaBoundOptimizer(learning_rate=learning_rate)
        elif(optimizer_name is "radam"):
            from keras_radam.training import RAdamOptimizer
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True,
                                                                 name = "exp_de"
                                                                 )
            optimizer = RAdamOptimizer(learning_rate=learning_rate_node)
           
        else:
            if not learning_rate is None:
                optimizer = tf.train.AdamOptimizer(learning_rate)
                learning_rate_node = tf.constant(0.0)
            else:
                learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                global_step=global_step,
                                                                decay_steps=decay_steps,
                                                                decay_rate=decay_rate,
                                                                staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node)
        if(optimizer_name is "adabound"):            
            optimizer =  optimizer.minimize(cost, global_step = global_step)
        else:
            optimizer =  optimizer.minimize(cost)


    return optimizer, learning_rate_node