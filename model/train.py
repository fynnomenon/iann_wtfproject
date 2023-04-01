import tensorflow as tf
import tqdm
import sys
from utils import generate_images

def train(model, train_ds, val_ds, epoch, args, train_summary_writer, val_summary_writer, hist, config_name, results_dir):
    '''
    Train the given model with the train_ds for training and val_ds for validation.

    Arguments: 
        model:tf.keras.Model
            Tensorflow model with a train_step and test_step method.
        train_ds:Dataset
            Dataset for training the model.
        val_ds:Dataset
            Dataset for validating the model.
        epoch:int
            Number of epochs to be performed. 
        args:argpase.Namespace
            Contains various training options (e.g. batch size or learning rate).
        train_summary_writer:SummaryWriter
            Summarizes the training metrics.
        val_summary_writer:SummaryWriter
            Summarizes the validation metrics.
        hist: dictionary
            saves the metrics for plotting
        config_name:str
            Represents the name of the configurations.
        results_dir:str
            Represents the path to the directory where the results will be saved.
    Returns:
        None
    '''
    # Training step
    idx = 0

    for data in tqdm.tqdm(train_ds, position = 0, leave = True):
        
        #get only first batch for visualization
        if idx == 0:
            visualize_train = data
            idx += 1
        
        metrics = model.train_step(data,args.text_with_dense)

        with train_summary_writer.as_default(): 
                for metric in model.metrics: 
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

    # Print the metrics and add to the history element
    for key, value in metrics.items():
        hist[f"train_{key}"].append(value.numpy())
        print(f"train_{key}: {value.numpy()}")

    #reset metric 
    model.reset_metrics()
    
    #generate the plot of each epoch for the training data
    generate_images(model, visualize_train, args, epoch, 'train', config_name, results_dir)
  
    # Validation step
    idx = 0
    for data in tqdm.tqdm(val_ds, position = 0, leave = True):
        if idx == 0:
            visualize_test = data
            idx += 1

        metrics = model.test_step(data,args.text_with_dense)

        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

    # Print the metrics and add to the history element 
    print("\n")
    for key, value in metrics.items():
        hist[f"test_{key}"].append(value.numpy())
        print(f"test_{key}: {value.numpy()}")

    #reset metric
    model.reset_metrics()
    print("\n")
    if args.save:
        model.save_weights(f"{results_dir}/saved_model"+"_" + args.model)
    
    #generate the plot of each epoch for the test data
    generate_images(model, visualize_test, args, epoch, 'test', config_name, results_dir)
    sys.stdout.flush()
    
    return hist