import os
import numpy as np
import tensorflow as tf
from .cost import get_cost
from .optimizer import get_optimizer
import time

class Trainer(object):
    """
   

    """
    def __init__(self, net,labels,gpu_device,path_weight, opt_kwargs={}, cost_kwargs={}):
        self.net = net
        self.labels = labels


        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        session_conf.gpu_options.visible_device_list = gpu_device
        session_conf.gpu_options.allow_growth = True
        self.sess = tf.Session(
#                config=tf.ConfigProto(log_device_placement=True)
                config=session_conf
                )

    def _initialize(self, batch_steps_per_epoch, labels):
        self.cost, self.accuracy = get_cost(
                                            # logits of self.net 
                                            labels, 
                                            self.cost_kwargs
                                            )

        self.optimizer, self.learning_rate_node = get_optimizer(self.cost, 
                                                                self.global_step,
                                                                batch_steps_per_epoch, 
                                                                self.opt_kwargs)
        print(self.cost, self.accuracy,"initial" )
        self.cost_summary = tf.summary.scalar(name = "cost_summary", tensor = self.cost)
        self.accuracy_summary = tf.summary.scalar(name = "accuracy_summary", tensor = self.accuracy)
        init = tf.global_variables_initializer()
        return init

    def train(self, data_provider, epochs=250,num_text_feature = 150,batch_steps_per_epoch = 1024,  is_restore = False):
        """

        """

        init = self._initialize(batch_steps_per_epoch, self.labels)
        saver = tf.train.Saver()


        log_dir = os.path.join(self.save_path,"logs")
        log_F1_dir = os.path.join(self.save_path,"log_F1")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(log_F1_dir):
            os.mkdir(log_F1_dir)



        writer = tf.summary.FileWriter(log_dir,self.sess.graph)     
        f =  open(os.path.join(log_F1_dir,str(int(time.time()))+"_log.txt"),"w+")
        self.sess.run(init)


        if(is_restore):
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(self.sess, ckpt.model_checkpoint_path) # restore all variables
        start = self.global_step.eval(session= self.sess) # get last global_step
        print("Start from:", start)



        for e in range(start,epochs):

            total_loss = 0
            total_acc = 0
            n = 0

            for data in (data_provider.list_train_data):

                _, loss, acc , cost_summary, accuracy_summary = self.sess.run([self.optimizer, self.cost, self.accuracy,self.cost_summary, self.accuracy_summary], 
                                             feed_dict={
                    # Some feed_dict
                    self.net.is_training: True
                }
                )

                total_acc += acc
                total_loss += loss
                n += 1


                writer.add_summary(cost_summary, e)
                writer.add_summary(accuracy_summary, e)

            if e % 5 == 0:
                print('train loss: ', total_loss /n, ' train acc: ', total_acc/n)
                for data in (data_provider.list_val_data):
                    loss, acc, out = self.sess.run([self.cost, self.accuracy, self.net.current_V], 
                                                    feed_dict={
                        # Some feed_dict
                        self.net.is_training: False
                    }
                    )
                
                ### Export val result here

                ###
                total_acc += acc
                total_loss += loss
                n += 1

            
                    
                ### Export some metric here
                f.write('cf: \n')
                f.write('{} val loss: {} , val acc: {}'.format(str(e) , total_loss /
                        n, total_acc/n))

                ###

                
                self.global_step.assign(e).eval(session= self.sess) 
                print(self.global_step.assign(e).eval(session= self.sess) )
                saver.save(self.sess, os.path.join(self.save_path ,"model.ckpt"),global_step = self.global_step, write_meta_graph=1) 
            








