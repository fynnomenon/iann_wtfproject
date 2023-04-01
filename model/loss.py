import tensorflow as tf
from tensorflow.keras.losses import Loss

class KLDivergence(Loss):
    """
    This function computes the Kullback-Leibler divergence between ground truth saliency
    maps (human fixation maps) and their predictions.
    """
    
    # Initialize instance attributes
    def __init__(self, eps=2.2204e-16):
        super(KLDivergence, self).__init__()
        self.eps=eps
        
    # Compute loss
    def call(self, y_pred, y_true):
        '''
        Computes the value of the KL Loss between the predicted and the ground truth saliency map.

        Arguments: 
            y_pred:tensor
                The predicted distribution for the saliency map. 
            y_true:tensor
                The ground truth distribution for the saliency map.
        Returns:
            loss value:tensor
                The loss value between the ground truth and predicted saliency map as scalar tensor.
        '''
        batch_size, w, h = y_pred.shape
        
        # Normalize the saliency map and ground truth to sum up to 1
        sum_y_true = tf.reduce_sum(tf.reshape(y_true, [batch_size, -1]), axis=1)
        sum_y_true = tf.reshape(sum_y_true, [batch_size, 1, 1])
        sum_y_true = tf.tile(sum_y_true, [1, w, h])
        y_true /= sum_y_true*1.0
        
        sum_y_pred = tf.reduce_sum(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        sum_y_pred = tf.reshape(sum_y_pred, [batch_size, 1, 1])
        sum_y_pred = tf.tile(sum_y_pred, [1, w, h])
        y_pred /= sum_y_pred*1.0
        
        # Flatten both maps
        y_true = tf.reshape(y_true,[batch_size, -1])
        y_pred = tf.reshape(y_pred, [batch_size, -1])
        
        # Compute kldiv between both distributions
        result = y_true * tf.math.log((y_true + self.eps) / (y_pred + self.eps))
        return tf.reduce_mean(tf.reduce_sum(result, axis=1))