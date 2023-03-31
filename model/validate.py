import tensorflow as tf
import tqdm
import sys
from utils import generate_images

def validate(model, val_ds, epoch, args, val_summary_writer, hist, config_name, results_dir):
    '''
    Validate the given model with the val_ds.

    Arguments: 
        model:tf.keras.Model
            Tensorflow model with a train_step and test_step method.
        val_ds:Dataset
            Dataset for validating the model.
        epoch:int
            Number of epochs to be performed. 
        args:argpase.Namespace
            Contains various training options (e.g. batch size or learning rate).
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
    # evaluation on validation set
    idx = 0
    for data in tqdm.tqdm(val_ds, position = 0, leave = True):
        if idx == 0:
            visualize_test = data
            idx += 1

        metrics = model.test_step(data)

        with val_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

    # print the metrics and add to history element
    print("\n")
    for key, value in metrics.items():
        hist[f"test_{key}"].append(value.numpy())
        print(f"test_{key}: {value.numpy()}")

    #reset metric
    model.reset_metrics()
    print("\n")
    if args.save:
        model.save_weights(f"{results_dir}/saved_model"+"_" + args.model)
    
    #generate the plot for each epoch
    generate_images(model, visualize_test, args, epoch, 'test', config_name, results_dir)
    sys.stdout.flush()
    
    return hist
