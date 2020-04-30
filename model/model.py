
import tensorflow as tf
import numpy as np

class TransformerGKV(object):
    """
    """
    def __init__(self,  num_class, model_kwargs={}):
        self.num_text_feature = model_kwargs.get('num_text_feature',300)
        self.num_class = num_class
        
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.sess = tf.Session()
        
        ### create model
        self.build_network()
        

    def build_network(self):
        self.layer_1()
        return self.V_in
        

    def create_input(self):
        self.V_in = tf.placeholder(dtype=tf.float32, shape=[None,
                                                            None, 
                                                            self.num_text_feature
                                                            ], name='input_vertices')

    def layer_1(self):
        self.V_in = V_in
        return V_in