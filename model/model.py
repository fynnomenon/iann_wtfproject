"""
@authors: faurand, chardes, ehagensieker
"""
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from metrics import AUC_Judd, CC, NSS, SIM, KLDiv
import sys

class BaselineModel(tf.keras.Model):
    '''
    Deep learning model to predict saliency maps of images. The model consists of a VGG-16 encoder, followed by a decoder
    consisting of a series of transpose convolution layers to upsample to obtain the saliency map.
    '''      
    def __init__(self,  fine_tune, use_pretrained, loss_function, optimizer,L2_norm = None, L1_norm = None):
        super(BaselineModel, self).__init__()
        '''
        The constructor of the model.

        Arguments: 
            fine_tune:int
                Specifies the number of layers of the encoder to fine-tune. 
            use_pretrained:boolean
                Indicates whether to use the pre-trained weights of the VGG-16 encoder or not.
            loss_function:loss function
                Computes the loss between the predicted and ground-truth saliency map.
            optimizer:optimizer
                Updates the weights of the model during training.
            l1_norm: kernel regularizer
                reduces complexity of the model, prevents overfitting
            l2_norm: kernel regularizer
                reduces complexity of the model, prevents overfitting
        '''
        
        self.loss_function = loss_function
        self.optimizer = optimizer
        #list of all metrics, we wanna use for evaluation
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.AUC(name="AUC"),
                             CC(),
                             NSS(),
                             KLDiv(),
                             SIM()]
        if L2_norm:
            kernel_regularizer=tf.keras.regularizers.L2(L2_norm)
        elif L1_norm:
            kernel_regularizer=tf.keras.regularizers.L1(L1_norm)
        else:
            kernel_regularizer=None
         
        # Specify whether to use the pretrained weights from ImageNet    
        if use_pretrained:
            self.vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        else:
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights=None)

        # Specify the number of layers to be fine-tuned (indicated from the last to the first layer)    
        if fine_tune > 0:
            for layer in self.vgg.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in self.vgg.layers:
                layer.trainable = False

        # Encoder: different blocks to use different outputs for concatination
        self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:4])
        self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[4:7])
        self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[7:11])
        self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[11:15])
        self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[15:])

        # Decoder: 
        self.linear_upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        
        self.deconv_layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3,strides=2, padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer2 = tf.keras.Sequential([           
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2,padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3,strides=2, padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same',activation='relu', use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same',activation='sigmoid', use_bias=True,kernel_regularizer=kernel_regularizer),
        ])

    def call(self, images, training=False):
        '''
        Perform operations on an input image to generate a saliency map.

        Arguments: 
            images: a tensor of shape (batch_size, height, width, channels) representing input images
            training: a boolean variable indicating whether the model is being trained or not
        Returns:
            x:tensor
                A tensor of shape representing the saliency map generated from the input images.
        Raises:
            AssertionError: 
                If the shape of the output tensor does not match the expected shape at any stage of the operation.
        '''
        batch_size = images.shape[0]
        
        out1 = self.conv_layer1(images,training = training)
        out2 = self.conv_layer2(out1,training = training)
        out3 = self.conv_layer3(out2,training = training)
        out4 = self.conv_layer4(out3,training = training)
        out5 = self.conv_layer5(out4,training = training)
        out5 = self.linear_upsampling(out5,training = training)
       
        x = tf.concat((out5, out4), axis=3)    
        x = self.deconv_layer1(x,training = training)

        x = tf.concat([x, out3], axis=3)
        x = self.deconv_layer2(x,training = training)

        x = tf.concat([x, out2], axis=3)
        x = self.deconv_layer3(x,training = training)

        x = tf.concat([x, out1], axis=3)
        x = self.deconv_layer4(x,training = training)
        x = self.deconv_layer5(x,training = training)

        assert x.shape == (batch_size, 224, 224, 1)
        x = tf.squeeze(x,axis = 3)
        assert x.shape == (batch_size, 224, 224)
     
        return x
    
    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()


    def train_step(self,data,_):
        '''
        Perform a training step on the model.

        Arguments: 
            data:tuple
                Contains the input images and the ground truth saliency map.
        Returns: 
            dictionary
                Contains the updated metrics for the model after the training step.
        Raises: 
            AssertionError:
                If the shape of the predicted saliency map does not match the shape of the ground truth map.
        '''
        img,gt = data

        with tf.GradientTape() as tape: 
            pred_map = self(img,training = True)
            assert pred_map.shape == gt.shape
            loss = self.loss_function(pred_map, gt)
            
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        

        self.metrics[0].update_state(loss)
        for i in range(1,6):
            self.metrics[i].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self,data,_):
        '''
        Perform a testing step on the model.

        Arguments: 
            data:tuple
                Contains the input images and the ground truth saliency map.
        Returns: 
            dictionary
                Contains the updated metrics for the model after the training step.
        Raises: 
            AssertionError:
                If the shape of the predicted saliency map does not match the shape of the ground truth map.
        '''
        img,gt = data
         
        pred_map = self(img,training = False)
        assert pred_map.shape == gt.shape
        loss = self.loss_function(pred_map, gt)
        
        self.metrics[0].update_state(loss)
        for i in range(1,6):
            self.metrics[i].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}


class MultimodalModel(tf.keras.Model):
    '''
    Deep learning model to predict saliency maps from images and image captions. 
    The model consists of a VGG-16 encoder, followed by a decoder
    consisting of a series of transpose convolution layers to obtain the saliency map. 
    Image captions are addionally added by concatenating them to the image embedding after the encoder.
    '''  
    def __init__(self, fine_tune, use_pretrained, loss_function, optimizer,L1_norm = None, L2_norm=None):
        super(MultimodalModel, self).__init__()
        '''
        The constructor of the model.

        Arguments: 
            fine_tune:int
                Specifies the number of layers of the encoder to be fine-tuned. 
            use_pretrained:boolean
                Indicates whether to use the pre-trained weights of the VGG-16 encoder or not.
            loss_function:loss function
                Computes the loss between the predicted and ground-truth saliency map.
            optimizer:optimizer
                Updates the weights of the model during training
        '''        
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.AUC(name="AUC"),
                             CC(),
                             NSS(),
                             KLDiv(),
                             SIM()]
        if L2_norm:
            kernel_regularizer=tf.keras.regularizers.L2(L2_norm)
        elif L1_norm:
            kernel_regularizer=tf.keras.regularizers.L1(L1_norm)
        else:
            kernel_regularizer=None
        
        # Specify whether to use the pretrained weights from ImageNet
        if use_pretrained:
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights='imagenet')
        else:
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights=None)
        
        # Specify the number of layers to be fine-tuned (indicated from the last to the first layer)    
        if fine_tune > 0:
            for layer in self.vgg.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in self.vgg.layers:
                layer.trainable = False
        
        # Encoder
        self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:4])
        self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[4:7])
        self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[7:11])
        self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[11:15])
        self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[15:])
        
        # Decoder
        self.linear_upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.deconv_layer1 = tf.keras.Sequential([           
            tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3,strides=2, padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2,padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3,strides=2, padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,padding='same', activation='relu',use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
        ])

        self.deconv_layer5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same',activation='relu', use_bias=True,kernel_regularizer=kernel_regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same',activation='sigmoid', use_bias=True,kernel_regularizer=kernel_regularizer),
        ])


        self.dense_layer = tf.keras.layers.Dense(196, activation='relu')
        self.reshape_layer = tf.keras.layers.Reshape((14, 14, 1))

    def call(self, input,dense=1,training=False):
        '''
        Perform operations on an input image to generate a saliency map.

        Arguments: 
            input:tuple     
                Contains images and their corresponding image captions.                
            dense:boolean
                Indicates whether the text embedding is reshaped through a dense layer and reshape layer
                or by expanding the dimensions and reshaping it to shape (1,14,14,1)
            training: a boolean variable indicating whether the model is being trained or not
        Returns:
            x:tensor
                A tensor of shape representing the saliency map generated from the input images.
        Raises:
            AssertionError: 
                If the shape of the output tensor does not match the expected shape at any stage of the operation.
        '''
        images,text_embedding = input
        batch_size = images.shape[0]

        out1 = self.conv_layer1(images,training=training)
        out2 = self.conv_layer2(out1,training=training)
        out3 = self.conv_layer3(out2,training=training)
        out4 = self.conv_layer4(out3,training=training)
        out5 = self.conv_layer5(out4,training=training)
        out5 = self.linear_upsampling(out5,training=training)

        if dense:
            text_embedding = self.dense_layer(text_embedding,training=training)
            text_embedding = self.reshape_layer(text_embedding,training=training)
        else:
            text_embedding = tf.expand_dims(text_embedding, 1)
            text_embedding = tf.tile(text_embedding, [1, 14, 14, 1])
        
        #concatinate the text embedding with the last, and fourth output
        x = tf.concat((out5, out4,text_embedding), axis=3)
        x = self.deconv_layer1(x,training=training)

        x = tf.concat([x, out3], axis=3)
        x = self.deconv_layer2(x,training=training)
     
        x = tf.concat([x, out2], axis=3)
        x = self.deconv_layer3(x,training=training)
        
        x = tf.concat([x, out1], axis=3)
        x = self.deconv_layer4(x,training=training)
        x = self.deconv_layer5(x,training=training)
  
        x = tf.image.resize(x, [224, 224])
        x = tf.squeeze(x,axis = 3)

        assert x.shape == (batch_size, 224, 224)
        return x
    
    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def train_step(self,data,text_with_dense):
        '''
        Perform a training step on the model.

        Arguments: 
            data:tuple
                Contains the input images and the ground truth saliency map.
        Returns: 
            dictionary
                Contains the updated metrics for the model after the training step.
        Raises: 
            AssertionError:
                If the shape of the predicted saliency map does not match the shape of the 
                ground truth map.
        '''

        input,gt = data

        with tf.GradientTape() as tape: 
            pred_map = self(input,text_with_dense,training = True)
            assert pred_map.shape == gt.shape
            loss = self.loss_function(pred_map, gt)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        for i in range(1,6):
            self.metrics[i].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self,data,text_with_dense):
        '''
        Perform a testing step on the model.

        Arguments: 
            data:tuple
                Contains the input images and the ground truth saliency map.
        Returns: 
            dictionary
                Contains the updated metrics for the model after the training step.
        Raises: 
            AssertionError:
                If the shape of the predicted saliency map does not match the shape of the 
                ground truth map.
        '''
        input,gt = data
        
        pred_map = self(input,text_with_dense,training = False)
        assert pred_map.shape == gt.shape 
        loss = self.loss_function(pred_map, gt)
       
        self.metrics[0].update_state(loss)
        for i in range(1,6):
            self.metrics[i].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}


