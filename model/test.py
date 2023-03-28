"""
@authors: faurand, chardes, ehagensieker
"""
import argparse
import glob
import os
import sys
import time
import pickle
from PIL import Image
#import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
#from dataloader import TestLoader, SaliconDataset
from loss import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir',default="../images/", type=str)
parser.add_argument('--model_val_path',default="../d_models/salicon_pnas.pt", type=str)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--results_dir',default="../results/", type=str)
parser.add_argument('--validate',default=0, type=int)
parser.add_argument('--save_results',default=1, type=int)
parser.add_argument('--dataset_dir',default="/home/samyak/old_saliency/saliency/SALICON_NEW/", type=str)

args = parser.parse_args()

device = tf.device('cuda' if tf.test.is_gpu_available() else 'cpu')

# what model to use: should we decide for one model                     
if args.enc_model == "pnas":
    print("PNAS Model")
    from model import PNASModel
    model = PNASModel()

# elif args.enc_model == "densenet":
#     print("DenseNet Model")
#     from model import DenseModel
#     model = DenseModel()

# elif args.enc_model == "resnet":
#     print("ResNet Model")
#     from model import ResNetModel
#     model = ResNetModel()
    
# elif args.enc_model == "vgg":
#     print("VGG Model")
#     from model import VGGModel
#     model = VGGModel()

elif args.enc_model == "mobilenet":
    print("Mobile NetV2")
    from model import MobileNetV2
    model = MobileNetV2()

if args.enc_model != "mobilenet":
    model = keras.utils.multi_gpu_model(model)
model.load_weights(args.model_val_path)


model = model.to(device)

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

val_img_dir = args.dataset_dir + "images/val/"
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixations/fixations/"

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = tf.data.Dataset.from_generator(
    lambda: val_dataset,
    output_types=(tf.float32, tf.float32, tf.float32)).batch(1)

def validate(model, loader, device, args):
    model.eval()
    tic = time.time()
    total_loss = 0.0
                    
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    
    for (img, gt, fixations) in tqdm(loader):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        gt = tf.convert_to_tensor(gt, dtype=tf.float32)
        fixations = tf.convert_to_tensor(fixations, dtype=tf.float32)

        # to device
        img = tf.convert_to_tensor(img).to(device)
        gt = tf.convert_to_tensor(gt).to(device)
        fixations = tf.convert_to_tensor(fixations).to(device)
        
        pred_map = model(img)

         # Blurring
        blur_map = tf.squeeze(pred_map).numpy()
        blur_map = blur(blur_map)
        blur_map = tf.convert_to_tensor(blur_map, dtype=tf.float32)
        
        cc_loss.update(cc(blur_map, gt))    
        kldiv_loss.update(kldiv(blur_map, gt))    
 

    print('CC : {:.5f}, KLDIV : {:.5f}  time:{:3f} minutes'.format(cc_loss.avg, kldiv_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()
    
    return cc_loss.avg

if args.validate:
    val_loader = tf.data.Dataset.from_generator(
        lambda: val_dataset,
        output_types=(tf.float32, tf.float32, tf.float32)
    ).batch(1)
    with tf.device(device):
        validate(model, val_loader, device, args)
        
if args.save_results:
	visualize_model(model, vis_loader, device, args)
