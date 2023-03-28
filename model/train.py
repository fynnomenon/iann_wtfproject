"""
@authors: faurand, chardes, ehagensieker
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
from scipy.stats import multivariate_normal
#from dataloader import SaliconDataset
#import the different loss functions
from loss import *
#import the blurring for smoothing the image and reducing the noise + AverageMeter to update the values
from utils import blur, AverageMeter
import pickle
import tensorflow.compat.v1 as tf_compat
from tensorflow.keras.utils import Sequence
import tensorflow.keras.initializers as init
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image



#all arguments that can be choosen/where we can choose between different options (can delete some losses)
parser = argparse.ArgumentParser()

parser.add_argument('--no_epochs',default=5, type=int)

parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=False, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--dilation',default=False, type=bool)
parser.add_argument('--enc_model',default="vgg", type=str)
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--load_weight',default=1, type=int)
parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=0, type=int)

parser.add_argument('--dataset_dir',default="../tfds_salicon", type=str)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)

parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--data',default='salicon', type=str)
parser.add_argument('--colab',default=True, type=bool)

#run parser and place extracted data in args
# 
# 
# 
# 
#  object
args = parser.parse_args()


#specify the directory of the dataset - construct the paths to the different directories
train_img_dir = args.dataset_dir + "images/train/"
train_gt_dir = args.dataset_dir + "maps/train/"
train_fix_dir = args.dataset_dir + "fixations/train/"

val_img_dir = args.dataset_dir + "images/val/"
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixations/fixations/"


def preprocessing(data,args):
    data = data.map(lambda img,_,sal: (img,tf.squeeze(sal,axis = 2)))
    data = data.batch(args.batch_size).shuffle(True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return data
def preprocessing_multi(data,args):
    data = data.map(lambda img,cap,sal: (img,(tf.cast(cap, dtype=tf.float32),tf.squeeze(sal,axis = 2))))
    #data = data.map(lambda img,cap,sal: (img,(tf.tile(tf.expand_dims(cap, 1),[1, 14, 14, 1]),tf.squeeze(sal,axis = 2))))
    data = data.batch(args.batch_size).shuffle(True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return data

if args.data == 'cap_gaze':
    if args.colab:
        dataset_path = '/content/drive/MyDrive/project_ann/tfds_capgaze1'
    else:
        dataset_path = '../tfds_capgaze1'
        

    print("Data: capgaze1")
    
    ds = tf.data.Dataset.load(dataset_path)
    ds = tf.data.Dataset.load('../tfds_capgaze1', compression='GZIP')
    train_loader = ds.skip(200)
    test_loader = ds.take(200)
    
    
else:
    if args.colab:
        dataset_path = '/content/drive/MyDrive/project_ann/tfds_salicon'
    else:
        dataset_path = args.dataset_dir

    print("Data: salicon")
    
    train_loader = tf.data.Dataset.load(dataset_path + '/train2014', compression='GZIP')
    test_loader = tf.data.Dataset.load(dataset_path + '/val2014', compression='GZIP')

    # train_loader = tf.data.Dataset.load('../tfds_salicon/train2014', compression='GZIP')
    # test_loader = tf.data.Dataset.load('../tfds_salicon/val2014', compression='GZIP')

# train_loader = train_loader.take(100)
# test_loader = test_loader.take(100)

    
if args.enc_model == "vgg":
    print("VGG Model")
    from model import VGGModel
    model = VGGModel(train_enc=bool(args.train_enc), load_weight=args.load_weight,loss_function= kldiv)
    train_loader = preprocessing(train_loader,args)
    test_loader = preprocessing(test_loader,args)

    multi = False


elif args.enc_model == "multimodal":
    print("Multimodal Model")
    from model import MultimodalModel
    model = MultimodalModel(train_enc=bool(args.train_enc), load_weight=args.load_weight,loss_function=tf.keras.losses.BinaryCrossentropy())
    train_loader = preprocessing_multi(train_loader,args)
    test_loader = preprocessing_multi(test_loader,args)
    multi = True

# if tf.test.is_gpu_available():
#     print("Use GPU")
#     devices = tf.coonfig.kist_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(devices[0], True)
# else: 
#     print("Use CPU")




def loss_func(pred_map, gt, args):
    loss = tf.constant(0.0, dtype = tf.float32)
    #criterion = tf.keras.losses.MeanAbsoluteError()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    # if args.cc:
    #     loss += args.cc_coeff * cc(pred_map, gt)
    return loss

def create_summary_writers(config_name):
    '''
    create the summary writer to have access to the metrics of the model 
    '''
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"logs/tests/{config_name}/{current_time}/train"
    val_log_path = f"logs/tests/{config_name}/{current_time}/val"

    # log writer
    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    val_summary_writer = tf.summary.create_file_writer(val_log_path)

    return train_summary_writer, val_summary_writer
   

def train(model, optimizer, train_ds, val_ds,epoch, args,train_summary_writer,val_summary_writer):
    #loader - loads batches of the data/provides batches of input data 
    #model.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0
    #idx = 0
    
    

    for (img, cap_gt) in tqdm.tqdm(train_ds, position = 0, leave = True):
    
    #for idx, (img, gt, fixations) in enumerate(loader):
        
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        if multi:
            cap = cap_gt[0]
            gt = cap_gt[1]
            cap = tf.convert_to_tensor(cap, dtype=tf.float32)
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            metrics = model.train_step(img,cap,gt)

        else: 
            gt = tf.convert_to_tensor(cap_gt, dtype=tf.float32)
            metrics = model.train_step(img,gt)
        #fixations = tf.convert_to_tensor(fixations, dtype=tf.float32)
        with train_summary_writer.as_default(): 
                for metric in model.metrics: 
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics and add to history element
    for key, value in metrics.items():
        print(f"train_{key}: {value.numpy()}")

    #reset metric 
    model.reset_metrics()

#evaluation on validation set
    for (img,cap_gt) in tqdm.tqdm(val_ds, position = 0, leave = True):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        if multi:
            cap = cap_gt[0]
            gt = cap_gt[1]
            cap = tf.convert_to_tensor(cap, dtype=tf.float32)
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            metrics = model.test_step(img, cap,gt)
        else: 
            gt = tf.convert_to_tensor(cap_gt, dtype=tf.float32)
            metrics = model.test_step(img,gt)

        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

    # print the metrics and add to history element
    for key, value in metrics.items():
        print(f"test_{key}: {value.numpy()}")

    #reset metric
    model.reset_metrics()
    print("\n")
    
    



        # if idx%args.log_interval==(args.log_interval-1):
        #     print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
        #     cur_loss = 0.0
        #     sys.stdout.flush()
    
    #print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()


def validate(model, loader, epoch, args):
    #model.eval()
    tic = time.time()
    total_loss = 0.0
    
    #cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    
    for (img,cap_gt) in tqdm.tqdm(loader, position = 0, leave = True):
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        if multi:
            cap = cap_gt[0]
            gt = cap_gt[1]
            cap = tf.convert_to_tensor(cap, dtype=tf.float32)
            gt = tf.convert_to_tensor(gt, dtype=tf.float32)
            pred_map = model(img, cap, training = False)
        else: 
            gt = tf.convert_to_tensor(cap_gt, dtype=tf.float32)
            pred_map = model(img, training = False)
        #fixations = tf.convert_to_tensor(fixations, dtype=tf.float32)
        
        

        # Blurring
        blur_map = tf.squeeze(pred_map).numpy()
        blur_map = blur(blur_map)
        blur_map = tf.convert_to_tensor(blur_map, dtype=tf.float32)
        
        #cc_loss.update(cc(blur_map, gt))    
        kldiv_loss.update(kldiv(blur_map, gt))    

    #print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}  time:{:3f} minutes'.format(epoch, 0,kldiv_loss.avg (time.time()-tic)/60))
    print('[{:2d},   val] KLDIV : {:.5f} time:{:3f} minutes'.format(epoch, kldiv_loss.avg.numpy(), (time.time()-tic)/60))
    sys.stdout.flush()
    
    return kldiv_loss.avg

#params = list(filter(lambda p: p.requires_grad, model.parameters())) 

#choose the optimizer you wanna test
if args.optim=="Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
if args.optim=="Adagrad":
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=args.lr)
if args.optim=="SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum = 0.9)
if args.lr_sched:
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr, decay_steps=args.step_size, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

for epoch in range(0, args.no_epochs):
    train_summary_writer, val_summary_writer = create_summary_writers(config_name = f'RUN')
    
    train(model, optimizer, train_loader,test_loader, epoch, args,train_summary_writer,val_summary_writer)
    
 

    # cc_loss = validate(model, test_loader, epoch, args)
    # if epoch == 0:
    #     best_loss = cc_loss.numpy()
    # if best_loss < cc_loss.numpy():
    #     best_loss = cc_loss.numpy()
    #     print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
    #     # if len(tf.distribute.get_strategy().extended.worker_devices) > 1:
    #     #     model.save_weights(args.model_val_path)
    #     # else:
    #     #     model.save_weights(args.model_val_path)
    #     model.save_weights(args.model_val_path)
    # else:
    #     print("no change in loss")
    # print("best loss: ", best_loss)
    # print()

    # if args.lr_sched:
    #     scheduler.step()

