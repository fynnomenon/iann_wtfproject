import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

def create_summary_writers(args, config_name, results_dir):
    '''
    Create tensorflow summary writers for the training and validation metrics.

    Arguments: 
        args:argparse.Namespace
            Contains various training options.
        config_name:str
            Represents the name of the configuration.
        results:dir:str
            Represents the path to the directory where the results will be saved.
    Returns:
        train_summary_writer:SummaryWriter
            Summary writer for training metrics.
        val_summary_writer:SummaryWriter
            Summary writer for validation metrics.
    '''
    train_log_path = f"{results_dir}/logs/train/{config_name}"
    val_log_path = f"{results_dir}/logs/val/{config_name}"

    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    val_summary_writer = tf.summary.create_file_writer(val_log_path)

    return train_summary_writer, val_summary_writer

def _deprocess_img(processed_img):
    '''
    Takes a preprocessed image used by VGG-16 and returns the corresponding original image. This is done
    by adding the mean pixel values, reversing the color channel back to RGB and clipping the values.
    Arguments:
    processed_img:tensor
    Preprocessed image in shape(1,224,224)
    Returns:
    img:tensor
    Original image in tf.uint8 format with shape(224,224,3).
    '''
    imagenet_means = [103.939, 116.779, 123.68]
    means = tf.reshape(tf.constant(imagenet_means), [1, 1, 3])
    img = processed_img + means
    img = tf.reverse(img, axis=[-1])
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, tf.uint8)

    return img

def generate_images(model, test_input, args, epoch, data, config_name, results_dir): 
    '''
    Generate and save images of the predicted saliency map for the input image and its ground truth.

    Arguments:
        model:tf.keras.Model
            The model used for prediction.
        test_input:tuple
            Input for the prediction and its ground truth.
        args:argpase.Namespace
            Entails information, like the model name
        epoch:int
            Represents the current epoch of the training.
        data:str
            Represents the data type (train vs test).
        config_name:str
            Represents the name of the configuration being used for the model.
        results_dir:str
            Represents the dirextory to save the images.
    Returns:
        None
    '''
    input, target = test_input
    pred = model(input, training = False)
    
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 6))
    for row in range(3):
        if args.model == 'multimodal':
            imgs, caps = input
            img = imgs[row]
        else:
            img = input[row]
    
        axes[row][0].imshow(pred[row], cmap='plasma')
        axes[row][0].set_title(str(row+1) + ". Predicted")
        
        axes[row][1].imshow(_deprocess_img(img))
        axes[row][1].imshow(pred[row], cmap='plasma', alpha=0.5, interpolation='bilinear')
        
        axes[row][2].imshow(target[row], cmap='plasma')
        axes[row][2].set_title(str(row+1) + ". Original")
        
        axes[row][3].imshow(_deprocess_img(img))
        axes[row][3].imshow(target[row], cmap='plasma', alpha=0.5, interpolation='bilinear')
        
        axes[row][0].axis('off')
        axes[row][1].axis('off')
        axes[row][2].axis('off')
        axes[row][3].axis('off')
    
    image_dir = f'{results_dir}/images/{config_name}'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    fig.tight_layout()
    plt.savefig(f"{image_dir}/epoch_{data}_{str(epoch)}")
    plt.close()
                                 
def save_hist(hist, config_name, results_dir):
    '''
    Saves a directory of the training history to a csv file.
    Arguments:
    hist:dict
    A dictionary containing the training history data.
    config_name:str
    The name of the configuration used to generate the training history data.
    results_dir:str
    The directory where the training history file should be saved.
    Returns:
    None
    '''
    hist_dir = f'{results_dir}/history'
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    
    pd.DataFrame.from_dict(hist).to_csv(f'{hist_dir}/{config_name}.csv', index=False)