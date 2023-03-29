"""
@authors: faurand, chardes, ehagensieker
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet169

import sys
# sys.path.append('../PNAS/')
# from PNASnet import *
# from genotypes import PNASNet

#various encoder for training the mdoel -> PNAS model best pretraining model regarding the paper
# encoder for feature extraction and decoder with deconvolutional layers 


class VGGModel(tf.keras.Model):
    
    def __init__(self, num_channels=3, train_enc=False, load_weight=1,loss_function=tf.keras.losses.KLDivergence(),optimizer = tf.keras.optimizers.Adam()):
        super(VGGModel, self).__init__()
        
        self.num_channels = num_channels
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                            tf.keras.metrics.KLDivergence(name="kl_divergence"),
                            tf.keras.metrics.MeanSquaredError(name = 'MSE')]
     
        
        
        if load_weight:
            #self.vgg.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights='imagenet') #include_top entfernt
        else:
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights=None)
            
        for layer in self.vgg.layers:
            layer.trainable = train_enc
            
        # self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:5])
        # self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[5:9])
        # self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[9:14])
        # self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[14:18])
        # self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[18:])

        self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:4])
        self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[4:7])
        self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[7:11])
        self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[11:15])
        self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[15:])


        self.linear_upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.deconv_layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, images, training=False):
        batch_size = images.shape[0]
        
        
        out1 = self.conv_layer1(images)
        
        #print(out1.shape)
        out2 = self.conv_layer2(out1)
        #print(out2.shape)
        out3 = self.conv_layer3(out2)
        #print(out3.shape)
        out4 = self.conv_layer4(out3)
        #print(out4.shape)
        out5 = self.conv_layer5(out4)
        #print(out5.shape)
        out5 = self.linear_upsampling(out5)
        # out4 = self.linear_upsampling(out4) #zusätzlich hinzugefügt weil sonst outputs nicht übereinstimmen (in pytorch schon, in tensorflow kommt anderer output raus)

        
        ###ab hier alle assertions angepasst, weil bei dem model andere shapes rauskommen###

       
        #assert out5.shape == (batch_size, 14, 14, 512)
       
        
        #print(out4.shape)
        #shape=(1, 128)
        x = tf.concat((out5, out4), axis=3)
       
        #assert x.shape == (batch_size, 14, 14, 1024)
        

        
        x = self.deconv_layer1(x)
        
      
       #assert x.shape == (batch_size, 28, 28, 512)
        
        #out3 = self.linear_upsampling(out3) #added to get same shape

        x = tf.concat([x, out3], axis=3)
       
        
        #assert x.shape == (batch_size, 28, 28, 1024)
        
        
        x = self.deconv_layer2(x)
      
        #assert x.shape == (batch_size, 56, 56, 256)
        
     
        x = tf.concat([x, out2], axis=3)
       
       
        #assert x.shape == (batch_size, 56, 56, 512)
        
        x = self.deconv_layer3(x)
        
        #assert x.shape == (batch_size, 112, 112, 128)
      
        
        x = tf.concat([x, out1], axis=3)
        #assert x.shape == (batch_size, 112, 112, 256)
       
        x = self.deconv_layer4(x)
       
      
        
        x = self.deconv_layer5(x)
        #assert x.shape == (batch_size, 256, 256, 1)
        assert x.shape == (batch_size, 224, 224, 1)
      
       
        x = tf.squeeze(x,axis = 3)
        #assert x.shape == (batch_size, 256, 256)
        assert x.shape == (batch_size, 224, 224)
     
        return x
    
    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()


    def train_step(self,data):
        img,gt = data
        with tf.GradientTape() as tape: 
            pred_map = self(img,training = True)
      
            assert pred_map.shape == gt.shape
            
            loss = tf.constant(0.0, dtype = tf.float32)
            loss += self.loss_function(pred_map,gt)
            #loss = self.loss_function(pred_map, gt)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(gt, pred_map)
        self.metrics[2].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self,data):
        img,gt = data
        
        
        pred_map = self(img,training = False)
    
        assert pred_map.shape == gt.shape
        
        loss = self.loss_function(pred_map, gt)
        
       
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(gt, pred_map)
        self.metrics[2].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}


class MultimodalModel(tf.keras.Model):
    
    def __init__(self, num_channels=3, train_enc=False, load_weight=1,loss_function=tf.keras.losses.KLDivergence(),optimizer = tf.keras.optimizers.Adam()):
        super(MultimodalModel, self).__init__()
        
        self.num_channels = num_channels
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                            tf.keras.metrics.KLDivergence(name="kl_divergence"),
                            tf.keras.metrics.MeanSquaredError(name = 'MSE')]
     
        
        
        if load_weight:
            #self.vgg.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights='imagenet') #include_top entfernt
        else:
            self.vgg = tf.keras.applications.VGG16(include_top=False,weights=None)
            
        for layer in self.vgg.layers:
            layer.trainable = train_enc
            
        # self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:5])
        # self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[5:9])
        # self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[9:14])
        # self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[14:18])
        # self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[18:])

        self.conv_layer1 = tf.keras.Sequential(self.vgg.layers[:4])
        self.conv_layer2 = tf.keras.Sequential(self.vgg.layers[4:7])
        self.conv_layer3 = tf.keras.Sequential(self.vgg.layers[7:11])
        self.conv_layer4 = tf.keras.Sequential(self.vgg.layers[11:15])
        self.conv_layer5 = tf.keras.Sequential(self.vgg.layers[15:])
        
        self.linear_upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.deconv_layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.Activation('sigmoid')
        ])

        self.dense_layer = tf.keras.layers.Dense(196, activation='relu')
        self.reshape_layer = tf.keras.layers.Reshape((14, 14, 1))

    def call(self, input,dense=True,training=False):
        images,text_embedding = input

        batch_size = images.shape[0]
        
        
        out1 = self.conv_layer1(images,training=training)
        
        #print(out1.shape)
        out2 = self.conv_layer2(out1,training=training)
        #print(out2.shape)
        out3 = self.conv_layer3(out2,training=training)
        #print(out3.shape)
        out4 = self.conv_layer4(out3,training=training)
        #print(out4.shape)
        out5 = self.conv_layer5(out4,training=training)
        #print(out5.shape)
        out5 = self.linear_upsampling(out5,training=training)
        
        # out4 = self.linear_upsampling(out4) #zusätzlich hinzugefügt weil sonst outputs nicht übereinstimmen (in pytorch schon, in tensorflow kommt anderer output raus)

        
        ###ab hier alle assertions angepasst, weil bei dem model andere shapes rauskommen###

        #assert out5.shape == (batch_size, 16, 16, 512)
        assert out5.shape == (batch_size, 14, 14, 512)
        
        #print(out4.shape)
        #shape=(1, 768)
        if dense:
            text_embedding = self.dense_layer(text_embedding,training=training)
            text_embedding = self.reshape_layer(text_embedding,training=training)
        else:
            text_embedding = tf.expand_dims(text_embedding, 1)
            text_embedding = tf.tile(text_embedding, [1, 14, 14, 1])
        #dense layer 

        #print("t-shape: ",text_embedding.shape)
        x = tf.concat((out5, out4,text_embedding), axis=3)
        #print("x-shape: ",x.shape)
        #assert x.shape == (batch_size, 16, 16, 1024)
        #assert x.shape == (batch_size, 14, 14, 1024)
        
        x = self.deconv_layer1(x,training=training)
        
        #assert x.shape == (batch_size, 32, 32, 512)
        #assert x.shape == (batch_size, 28, 28, 512)
        #out3 = self.linear_upsampling(out3) #added to get same shape

        x = tf.concat([x, out3], axis=3)
        #assert x.shape == (batch_size, 32, 32, 1024)
        #assert x.shape == (batch_size, 28, 28, 1024)
        x = self.deconv_layer2(x,training=training)
        
        #assert x.shape == (batch_size, 64, 64, 256)
        #assert x.shape == (batch_size, 56, 56, 256)
     
        x = tf.concat([x, out2], axis=3)
       
        #assert x.shape == (batch_size, 64, 64, 512)
        #assert x.shape == (batch_size, 56, 56, 512)
        x = self.deconv_layer3(x,training=training)
        #assert x.shape == (batch_size, 128, 128, 128)
        #assert x.shape == (batch_size, 112, 112, 128)
        
        x = tf.concat([x, out1], axis=3)
        #assert x.shape == (batch_size, 128, 128, 256)
        #assert x.shape == (batch_size, 112, 112, 256)
        x = self.deconv_layer4(x,training=training)
        x = self.deconv_layer5(x,training=training)
        #assert x.shape == (batch_size, 256, 256, 1)
       # assert x.shape == (batch_size, 224, 224, 1)
       
        
        x = tf.image.resize(x, [224, 224])
        x = tf.squeeze(x,axis = 3)

        #print(x.shape)
        #assert x.shape == (batch_size, 256, 256)
        assert x.shape == (batch_size, 224, 224)
        return x
    
    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()


    def train_step(self,data):
        input,gt = data
        with tf.GradientTape() as tape: 
            pred_map = self(input,training = True)
      
            assert pred_map.shape == gt.shape
            
            loss = tf.constant(0.0, dtype = tf.float32)
            loss += self.loss_function(pred_map,gt)
            #loss = self.loss_function(pred_map, gt)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(gt, pred_map)
        self.metrics[2].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self,data):
        input,gt = data
        
        pred_map = self(input,training = False)
    
        assert pred_map.shape == gt.shape
        
        loss = self.loss_function(pred_map, gt)
        
       
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(gt, pred_map)
        self.metrics[2].update_state(gt, pred_map)

        return {m.name : m.result() for m in self.metrics}





class MobileNetV2(tf.keras.Model):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(MobileNetV2, self).__init__()

        #tensorflow application that instantiates the mobilenetv2 architecture
      
        self.mobilenet = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
        
        for layer in self.mobilenet.layers:
            layer.trainable = train_enc

        self.linear_upsampling = tf.keras.layers.UpSampling2D(size = 2, interpolation = 'bilinear')
        self.conv_layer1 = tf.keras.Sequential(self.mobilenet.layers[:2])
        self.conv_layer2 = tf.keras.Sequential(self.mobilenet.layers[2:4])
        self.conv_layer3 = tf.keras.Sequential(self.mobilenet.layers[4:7])
        self.conv_layer4 = tf.keras.Sequential(self.mobilenet.layers[7:14])
        self.conv_layer5 = tf.keras.Sequential(self.mobilenet.layers[14:]) #test:18 weg


        self.deconv_layer0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 96, kernel_size=3, padding='same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])

        self.deconv_layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 24, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            self.linear_upsampling
        ])
        self.deconv_layer5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, padding = 'same', use_bias = True),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, images):
        print(len(self.mobilenet.layers))
        print("Call_method_mobile")
        batch_size = images.shape[0]
        print(images.shape)
        print("shape_problem")
        out1 = self.conv_layer1(images)
        print(out1.shape)
        print("first_conv_done")
        out2 = self.conv_layer2(out1)
        print(out2.shape)
        print("2_conv_done")
        out3 = self.conv_layer3(out2)
        print(out3.shape)
        print("3_conv_done")
        out4 = self.conv_layer4(out3)
        print(out4.shape)
        print("4_conv_done")
        out5 = self.conv_layer5(out4)
        print("no_model_problems")
        
        
        assert out1.shape == (batch_size, 128, 128, 16)
        print("first_assertion done")
        assert out2.shape == (batch_size, 64, 64, 24)
        assert out3.shape == (batch_size, 32, 32, 32)
        assert out4.shape == (batch_size, 16, 16, 96)
        assert out5.shape == (batch_size, 8, 8, 1280)
        print("no_shape_problems")
        out5 = self.deconv_layer0(out5)

        x = tf.concat([out5,out4], axis = 3)
        x = self.deconv_layer1(x)

        x = tf.concat([x,out3],axis = 3)
        x = self.deconv_layer2(x)

        x = tf.concat([x,out2], axis = 3)
        x = self.deconv_layer3(x)
        
        x = tf.concat([x,out1], axis = 3)
        x = self.deconv_layer4(x)
        x = self.deconv_layer5(x)
        x = tf.squeeze(x,axis = 3)
        return x
    
    