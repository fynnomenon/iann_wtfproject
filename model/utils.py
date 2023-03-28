import tensorflow as tf
import cv2
import os
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def blur(img):
    '''
    image smoothing to reduce the noise: sharp edges in images are smoothed while minimizing too much blurring
    '''
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return tf.convert_to_tensor(bl)

def plot(pred, gt, args, idx):
    '''
    plot the predicted map and the ground truth for comparison 
    '''
    #tf.transpose to convert the tensor into numpy array with dim (height, width, channels)
    pred_npimg = np.transpose(pred.cpu().numpy(), (1, 2, 0))
    gt_npimg = np.transpose(gt.cpu().numpy(), (1, 2, 0))

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(gt_npimg)
    ax[0].set_title("Original")

    ax[1].imshow(pred_npimg)
    ax[1].set_title("Predicted")

    plt.savefig(join(args.results_dir, '{}_{}.png'.format(epoch, idx+1)))
    plt.close()

def visualize_model(model, loader, device, args):
    '''
    visualize the preformance?? 
    '''
    with tf.device(device):
        model.eval()
        os.makedirs(args.results_dir, exist_ok=True)
        
        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)
            
            pred_map = model(img)
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (sz[0], sz[1]))
            
            pred_map = blur(pred_map)
            img_save(pred_map, join(args.results_dir, img_id[0]), normalize=True)

def img_save(tensor, fp, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    '''
    save the images 
    '''
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range, scale_each=scale_each)

    ndarr = np.round(grid.numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
    ndarr = np.transpose(ndarr, (1, 2, 0))
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten == "png":
        im.save(fp, format=format, compress_level=0)
    else:
        im.save(fp, format=format, quality=100) #for jpg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def im2heat(pred_dir, a, gt, exten='.png'):
    pred_nm = pred_dir + a + exten
    #load the image in the program; returns ndarray
    pred = cv2.imread(pred_nm, 0)
    #get pseudocolored image
    heatmap_img = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    heatmap_img = convert(heatmap_img)
    pred = np.stack((pred, pred, pred),2).astype('float32')
    pred = pred / 255.0
    
    return np.uint8(pred * heatmap_img + (1.0-pred) * gt)

def convert(image):
    return tf.image.convert_image_dtype(image, dtype=tf.float32)