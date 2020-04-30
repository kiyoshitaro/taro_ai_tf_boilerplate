import tensorflow as tf
import numpy as np


def create_class_weights(labels,all_label, kwargs):
    is_class_weights = kwargs.get('class_weights', False)
    type_weight = kwargs.get('type_weight', 'inv_max')
    
    concat_lb = np.array([])
    cnt = np.zeros(int(max(np.amax(d) for d in all_label))+1)   
    
    for doc_lb in all_label:
        concat_lb = np.append(concat_lb,doc_lb)
        for lb in doc_lb:
            cnt[int(lb)] = cnt[int(lb)] + 1

    if(is_class_weights):
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', np.unique(concat_lb), concat_lb)
    #        class_weights = tf.constant(cnt) # shape (, num_classes)
        return class_weights
    
    if(type_weight == 'rem_sum'):
        rot = [i/np.sum(cnt) for i in cnt]
    elif(type_weight == 'rem_max'):
        rot  = [i/np.max(cnt) for i in cnt]
    elif(type_weight == "inv_sum"):
        rot = [np.max(cnt)/i for i in cnt]
    else:
        rot = [np.max(cnt)/i for i in cnt]
        max_val = max([d for d in rot if d != float("inf")])
        for i,val in enumerate(rot):
            if(val == float("inf")):
                rot[i] = max_val
        rot[1:] = [d/50 for d in rot[1:]]
    
    weights = tf.gather(rot, labels)   
    return weights

    
def get_cost(logits, labels, kwargs={}):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    is_weight: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    is_weight = kwargs.get('is_weight', False)
    with tf.variable_scope('loss') as scope:
        
        print(logits,"logit")
        
        # WRITE LOSS FUNCT HERE

        labels = tf.cast(labels, tf.int64)
        cross_entropy =  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        
        if(is_weight):
            all_label = kwargs.get('all_label', None)
            weights = create_class_weights(labels,all_label, kwargs = {})           
            labels = tf.cast(labels, tf.int64)          
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels,weights = weights)
    
        loss = tf.reduce_mean(cross_entropy)
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits,2),labels), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    return loss, accuracy

