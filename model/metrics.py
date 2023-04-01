import tensorflow as tf
import numpy as np
from loss import KLDivergence

class KLDiv(tf.keras.metrics.Mean):
    '''
    Create a custom Kullback-Leibler Divergence metric computed between the predicted and the ground truth distribution.
    '''
    def __init__(self):
        """
        Initializes a new instance of the `KLDiv` class.
        """
        super().__init__(name='KLDiv')

    def update_state(self, y_true, y_pred):
        """
        Accumulates the KL divergence between the predicted distribution and the true distribution.

        Arguments:
            y_true:tf.Tensor
                The true distribution, represented as a `tf.Tensor`.
            y_pred:tensor
                The predicted distribution, represented as a `tf.Tensor`.

        Returns:
            None
        """
        kl_divergence = KLDivergence()
        kldiv = kl_divergence(y_pred, y_true)
        
        super().update_state(kldiv)
     
class CC(tf.keras.metrics.Mean):
    '''
    Computes the Pearson Cross Correlation coefficient (CC) between the predicted and ground truth saliency map.
    It computes the correlation between the normalized versions of both distributions. 
    '''
    def __init__(self):
        '''
        Initializes a new instance of the `CC` class.
        '''
        super().__init__(name='CC')

    def update_state(self, y_true, y_pred):
        '''
        Accumulates the CC value for a batch of predictions and ground truths.

        Arguments:
            y_true:tensor
                The ground truth saliency maps of shape (batch_size, height, width).
            y_pred:tensor 
                The predicted saliency maps of shape (batch_size, height, width).
        Returns:
            None
        '''
        batch_size, w, h = y_pred.shape

        # Normalizes the saliency map and ground truth to have zero mean and unit std
        mean_y_true = tf.reduce_mean(tf.reshape(y_true, [batch_size, -1]), axis=1)
        mean_y_true = tf.tile(tf.reshape(mean_y_true, [batch_size, 1, 1]), [1, w, h])
        std_y_true = tf.math.reduce_std(tf.reshape(y_true, [batch_size, -1]), axis=1)
        std_y_true = tf.tile(tf.reshape(std_y_true, [batch_size, 1, 1]), [1, w, h])
        y_true_norm = (y_true - (mean_y_true*1.0)) / (std_y_true*1.0)
        
        mean_y_pred = tf.reduce_mean(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        mean_y_pred = tf.tile(tf.reshape(mean_y_pred, [batch_size, 1, 1]), [1, w, h]) 
        std_y_pred = tf.math.reduce_std(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        std_y_pred = tf.tile(tf.reshape(std_y_pred, [batch_size, 1, 1]), [1, w, h]) 
        y_pred_norm = (y_pred - (mean_y_pred*1.0)) / (std_y_pred*1.0)

        # Compute the covariance and variances
        cov = tf.reduce_sum(tf.reshape(y_pred_norm*y_true_norm, [batch_size, -1]), axis=1)
        var_pred = tf.reduce_sum(tf.reshape(tf.square(y_pred_norm), [batch_size, -1]), axis=1)
        var_true = tf.reduce_sum(tf.reshape(tf.square(y_true_norm), [batch_size, -1]), axis=1)
        
        # Compute the correlation coefficients
        cc = tf.reduce_mean(cov/tf.sqrt(var_pred * var_true))
        
        super().update_state(cc)

class NSS(tf.keras.metrics.Mean):
    '''
    Computes the Normalized Scanpath Saliency (NSS) between the predicted and ground truth saliency map.
    It calculates the mean of the normalized saliency map values at fixation locations.
    (Larger value implies better performance.)
    '''
    def __init__(self, eps=2.2204e-16):
        '''
        Initializes a new instance of the `NSS` class.
        Arguments: 
            eps:float, optional
                A small value added to the standard deviation to avoid division by zero.
        '''
        super().__init__(name='NSS')
        self.eps = eps

    def update_state(self, y_true, y_pred):
        '''
        Accumulates the (NSS) value for a batch of predictions and ground truths.

        Arguments:
            y_true:tensor
                The ground truth saliency maps of shape (batch_size, height, width).
            y_pred:tensor 
                The predicted saliency maps of shape (batch_size, height, width).
        Returns:
            None
        '''
        batch_size, w, h = y_pred.shape       
        
        # Discretize the ground truth
        y_true = tf.round(y_true)

        # Get the indices of the pixels in the ground truth that are 1 (pixels with a human fixation)
        indices = tf.where(tf.equal(y_true, 1))
        x = indices[:, 0]
        y = indices[:, 1]
        
        # Normalizes the saliency map to have zero mean and unit std
        mean_y_pred = tf.reduce_mean(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        mean_y_pred = tf.tile(tf.reshape(mean_y_pred, [batch_size, 1, 1]), [1, w, h]) 
        std_y_pred = tf.math.reduce_std(tf.reshape(y_pred, [batch_size, -1]), axis=1)
        std_y_pred = tf.tile(tf.reshape(std_y_pred, [batch_size, 1, 1]), [1, w, h])
        y_pred_norm = (y_pred - (mean_y_pred*1.0)) / ((std_y_pred*1.0) + self.eps)
        
        # Get the mean of the normalized saliency map values at the indices of the ground truth pixels with a human fixation
        nss = tf.reduce_mean(tf.gather_nd(y_pred_norm, tf.stack([x, y], axis=1)))
        
        super().update_state(nss)
    
class SIM(tf.keras.metrics.Mean):
    '''
    Computes the Similarity metric between the predicted and ground truth saliency map. The similarity is based on the intersection between
    the both. 
    '''
    def __init__(self):
        '''
        Initializes a new instance of the `SIM` class.
        '''
        super().__init__(name='SIM')

    def update_state(self, y_true, y_pred):
        '''
        Accumulates the Similarity for the predicted and ground truth saliency map.

        Arguments:
            y_true:tensor
                The ground truth saliency maps of shape (batch_size, height, width).
            y_pred:tensor 
                The predicted saliency maps of shape (batch_size, height, width).
        Returns:
            None
        '''
        batch_size, w, h = y_pred.shape
    
        # Normalize the saliency map and ground truth to sum up to 1
        sum_y_true = tf.reduce_sum(tf.reshape(y_true, [batch_size, -1]), axis=1, keepdims=True)
        sum_y_true = tf.reshape(sum_y_true, [batch_size, 1, 1])
        sum_y_true = tf.tile(sum_y_true, [1, w, h])
        y_true /= sum_y_true*1.0
        
        sum_y_pred = tf.reduce_sum(tf.reshape(y_pred, [batch_size, -1]), axis=1, keepdims=True)
        sum_y_pred = tf.reshape(sum_y_pred, [batch_size, 1, 1])
        sum_y_pred = tf.tile(sum_y_pred, [1, w, h])
        y_pred /= sum_y_pred*1.0
        
        # Flatten both maps
        y_true = tf.reshape(y_true,[batch_size, -1])
        y_pred = tf.reshape(y_pred, [batch_size, -1])
        
        # Calculate the similarity
        sim = tf.reduce_mean(tf.reduce_sum(tf.minimum(y_pred, y_true), axis=1))
    
        super().update_state(sim)

class AUC_Judd(tf.keras.metrics.Mean):
    '''
    Calculates the Area-Under-The-Curve-Judd (AUC_Judd) metric for evaluating the performance of a saliency model.
    '''
    def __init__(self):
        '''
        Initializes a new instance of the `AUC_Judd` class.
        '''
        super().__init__(name='AUC')

    def update_state(self, y_true, y_pred):
        '''
        Accumulates the AUC_Judd value for a batch of predictions and ground truths.

        Arguments:
            y_true:tensor
                The ground truth saliency maps of shape (batch_size, height, width).
            y_pred:tensor 
                The predicted saliency maps of shape (batch_size, height, width).
        Returns:
            None
        '''
        # Convert tensors to numpy arrays
        saliency_map = y_pred.numpy()
        fixation_map = y_true.numpy()

        # Jitter the saliency map slightly to disrupt ties of the same saliency value
        saliency_map += saliency_map + np.random.random(np.shape(saliency_map))*1e-7

        # Flatten both maps
        S = saliency_map[0,:,:].flatten()
        F = fixation_map[0,:,:].flatten()

        # Thresholds are calculated from the salience map, only at places where fixations are present
        sal_fix = S[F>0]
        thresholds = sorted(sal_fix, reverse=True)

        n_fix = len(sal_fix)
        n_pixels = len(S)

        tp = np.zeros(len(thresholds)+2)
        tp[0] = 0
        tp[-1] = 1

        fp = np.zeros(len(thresholds)+2)
        fp[0] = 0
        fp[-1] = 1
        
        # Calculate the true positive and false positive rates at each threshold
        for i, thresh in enumerate(thresholds):
            above_th = np.sum(S >= thresh) # Number of saliency map values above threshold
            tp[i+1] = (i + 1.0) / n_fix # Ratio saliency map values at fixation locations above threshold
            fp[i+1] = (above_th - i - 1.0) / (n_pixels - n_fix) # Ratio other saliency map values above threshold
        
        # Calculate the AUC-Judd using the trapezoidal rule
        aud_judd = np.trapz(tp, fp)
        
        super().update_state(tf.convert_to_tensor(aud_judd, dtype=tf.float32))
