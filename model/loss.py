"""
@authors: faurand, chardes, ehagensieker
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import tqdm
import pprint
import cv2

#%load_ext tensorboard
from PIL import Image

#different loss functions 
#should use kldiv and cc loss function as the paper mentions that it performs best

def kldiv(s_map, gt):
    '''
    compare two probability distributions: saliency map with ground truth 
    '''
    batch_size = s_map.shape[0]
    w = s_map.shape[1]
    h = s_map.shape[2]

    #sum along the batch, width and height dimension
    
    sum_s_map = tf.math.reduce_sum(tf.reshape(s_map, [batch_size, -1]), axis = 1)
    expand_s_map = tf.tile(tf.reshape(sum_s_map, [batch_size, 1, 1]), [1, w, h])
    
    assert expand_s_map.shape == s_map.shape

    sum_gt = tf.math.reduce_sum(tf.reshape(gt, [batch_size, -1]), axis = 1)
    expand_gt = tf.tile(tf.reshape(sum_gt, [batch_size, 1, 1]),[1, w, h])
    
    assert expand_gt.shape == s_map.shape

    #normalizes both distributions by dividing by the sum
    
    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = tf.reshape(s_map,[batch_size, -1])
    gt = tf.reshape(gt, [batch_size, -1])

    #kl div between both distributions
    
    eps = 2.2204e-16
    result = gt * tf.math.log(eps + gt/(s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    # loss value
    return tf.math.reduce_mean(tf.math.reduce_sum(result, 1))

# def kldiv(s_map, gt):
#     batch_size = tf.shape(s_map)[0]
#     w = tf.shape(s_map)[1]
#     h = tf.shape(s_map)[2]

#     sum_s_map = tf.reduce_sum(tf.reshape(s_map, (batch_size, -1)), axis=1)
#     #print("1.: ", sum_s_map.shape)
#     expand_s_map = tf.expand_dims(tf.expand_dims(sum_s_map, axis=1), axis=2)
#     #print("2.: ",expand_s_map.shape)
#     expand_s_map = tf.tile(expand_s_map, [1, w, h])


#     # print(s_map.shape)
#     # print(expand_s_map.shape)
    
#     assert expand_s_map.shape == s_map.shape

#     sum_gt = tf.reduce_sum(tf.reshape(gt, (batch_size, -1)), axis=1)
#     expand_gt = tf.expand_dims(tf.expand_dims(sum_gt, axis=1), axis=2)
#     expand_gt = tf.tile(expand_gt, [1, w, h])
    
#     assert expand_gt.shape == gt.shape

#     s_map = s_map/(expand_s_map*1.0)
#     gt = gt / (expand_gt*1.0)

#     s_map = tf.reshape(s_map, (batch_size, -1))
#     gt = tf.reshape(gt, (batch_size, -1))

#     eps = 2.2204e-16
#     result = gt * tf.math.log(eps + gt/(s_map + eps))
#     # print(tf.math.log(eps + gt/(s_map + eps))   )
#     return tf.reduce_mean(tf.reduce_sum(result, axis=1))



# def cc(s_map, gt):
#     batch_size = tf.shape(s_map)[0]
#     w = tf.shape(s_map)[1]
#     h = tf.shape(s_map)[2]
    
#     #pearson correlation coefficient: stddv and mean along batch, width and height
    
#     mean_s_map = tf.expand_dims(tf.reduce_mean(tf.reshape(s_map, [batch_size, -1]), axis=1), axis=[1, 2])
#     std_s_map = tf.expand_dims(tf.math.reduce_std(tf.reshape(s_map, [batch_size, -1]), axis=1), axis=[1, 2])

#     mean_gt = tf.expand_dims(tf.reduce_mean(tf.reshape(gt, [batch_size, -1]), axis=1), axis=[1, 2])
#     std_gt = tf.expand_dims(tf.math.reduce_std(tf.reshape(gt, [batch_size, -1]), axis=1), axis=[1, 2])

#     #standardizes by subtracting mean and dividing by stddv -> lin relationship between both distributions
    
#     s_map = (s_map - mean_s_map) / std_s_map
#     gt = (gt - mean_gt) / std_gt

#     ab = tf.reduce_sum(tf.reshape(s_map * gt, [batch_size, -1]), axis=1)
#     aa = tf.reduce_sum(tf.reshape(s_map * s_map, [batch_size, -1]), axis=1)
#     bb = tf.reduce_sum(tf.reshape(gt * gt, [batch_size, -1]), axis=1)
    
#     #loss value
#     return tf.reduce_mean(ab / (tf.sqrt(aa* bb)))

def cc(s_map, gt):
    batch_size = tf.shape(s_map)[0]
    w = tf.shape(s_map)[1]
    h = tf.shape(s_map)[2]

    mean_s_map = tf.expand_dims(tf.math.reduce_mean(tf.reshape(s_map, [batch_size, -1]), axis=1), axis=(1, 2))
    mean_s_map = tf.broadcast_to(mean_s_map, [batch_size, w, h])
    std_s_map = tf.expand_dims(tf.math.reduce_std(tf.reshape(s_map, [batch_size, -1]), axis=1), axis=(1, 2))
    std_s_map = tf.broadcast_to(std_s_map, [batch_size, w, h])

    mean_gt = tf.expand_dims(tf.math.reduce_mean(tf.reshape(gt, [batch_size, -1]), axis=1), axis=(1, 2))
    mean_gt = tf.broadcast_to(mean_gt, [batch_size, w, h])
    std_gt = tf.expand_dims(tf.math.reduce_std(tf.reshape(gt, [batch_size, -1]), axis=1), axis=(1, 2))
    std_gt = tf.broadcast_to(std_gt, [batch_size, w, h])

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = tf.math.reduce_sum(tf.reshape(s_map * gt, [batch_size, -1]), axis=1)
    aa = tf.math.reduce_sum(tf.reshape(s_map * s_map, [batch_size, -1]), axis=1)
    bb = tf.math.reduce_sum(tf.reshape(gt * gt, [batch_size, -1]), axis=1)

    return tf.math.reduce_mean(ab / (tf.math.sqrt(aa * bb)))

def auc_judd(saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if saliencyMap.size() != fixationMap.size():
        saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
        saliencyMap = tf.Tensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
        # saliencyMap = saliencyMap.cuda()
        # fixationMap = fixationMap.cuda()
    if len(saliencyMap.size())==3:
        saliencyMap = saliencyMap[0,:,:]
        fixationMap = fixationMap[0,:,:]
    saliencyMap = saliencyMap.numpy()
    fixationMap = fixationMap.numpy()
    if normalize:
        saliencyMap = normalize_map(saliencyMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score

def auc_shuff(s_map,gt,other_map,splits=100,stepsize=0.1):

    if len(s_map.size())==3:
        s_map = s_map[0,:,:]
        gt = gt[0,:,:]
        other_map = other_map[0,:,:]
    

    s_map = s_map.numpy()
    s_map = normalize_map(s_map)
    gt = gt.numpy()
    other_map = other_map.numpy()

    num_fixations = np.sum(gt)

    x,y = np.where(other_map==1)
    other_map_fixs = []
    for j in zip(x,y):
        other_map_fixs.append(j[0]*other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind==np.sum(other_map), 'something is wrong in auc shuffle'


    num_fixations_other = min(ind,num_fixations)

    num_pixels = s_map.shape[0]*s_map.shape[1]
    random_numbers = []
    for i in range(0,splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k%s_map.shape[0]-1, int(k/s_map.shape[0])])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0,0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map>=thresh] = 1.0
            num_overlap = np.where(np.add(temp,gt)==2)[0].shape[0]
            tp = num_overlap/(num_fixations*1.0)

            #fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map>thresh)[0])/(num_fixations*1.0)

            area.append((round(tp,4),round(fp,4)))

        area.append((1.0,1.0))
        area.sort(key = lambda x:x[0])
        tp_list =  [x[0] for x in area]
        fp_list =  [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))

    return np.mean(aucs)
