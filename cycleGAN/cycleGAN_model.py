import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()
from PIL import Image
import numpy as np 

from architecture import *

class CycleGAN:
    
    def __init__(self, input_shape,pool_size,beta1):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.lr_rate = tf.compat.v1.placeholder(dtype=tf.float32,shape=[],name="lr_rate")
        self.input_A = tf.placeholder(dtype=tf.float32, shape=[1, input_shape[0], input_shape[1], input_shape[2]], name="input_A")
        self.input_B = tf.placeholder(dtype=tf.float32, shape=[1, input_shape[0], input_shape[1], input_shape[2]], name="input_B")         
        self.fake_pool_Aimg = tf.placeholder(dtype=tf.float32, shape=[1, input_shape[0], input_shape[1], input_shape[2]], name="fake_pool_Aimg") 
        self.fake_pool_Bimg = tf.placeholder(dtype=tf.float32, shape=[1, input_shape[0], input_shape[1], input_shape[2]], name="fake_pool_Bimg") 
        
        self.gen_A,self.gen_B,self.cyclicA,self.cyclicB,self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B = self.model_arc(self.input_A,self.input_B,self.fake_pool_Aimg,self.fake_pool_Bimg)
        
        
        self.gen_loss_A, self.disc_loss_A, self.gen_loss_B,self.disc_loss_B = self.model_loss(self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B,self.input_A,self.cyclicA,self.input_B,self.cyclicB)
        
        
        self.discA_opt,self.discB_opt,self.genA_opt,self.genB_opt = self.model_opti(self.gen_loss_A,self.disc_loss_A,self.gen_loss_B,self.disc_loss_B,self.lr_rate,beta1)
        
    
    def model_arc(self,input_A,input_B,fake_pool_A,fake_pool_B):
        
        with tf.compat.v1.variable_scope("CycleGAN") as scope:
            gen_B = generator(input_A,name="generator_A")
            gen_A = generator(input_B,name="generator_B")
            real_disc_A = discriminator(input_A,name="discriminator_A")
            real_disc_B = discriminator(input_B,name="discriminator_B")
            
            scope.reuse_variables()
            
            fake_disc_A = discriminator(gen_A,name="discriminator_A")
            fake_disc_B = discriminator(gen_B,name="discriminator_B")
            
            cyclicA = generator(gen_B,name="generator_B")
            cyclicB = generator(gen_A,name="generator_A")
            
            scope.reuse_variables()
            
            fake_pool_disc_A = discriminator(fake_pool_A,name="discriminator_A")
            fake_pool_disc_B = discriminator(fake_pool_B,name="discriminator_B")
            
            return gen_A,gen_B,cyclicA,cyclicB,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B
        
    
    def model_loss(self,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B,input_A,cyclicA,input_B,cyclicB):
        
        cyclic_loss = tf.reduce_mean(tf.abs(input_A-cyclicA)) + tf.reduce_mean(tf.abs(input_B - cyclicB))
        
        disc_loss_A = tf.reduce_mean(tf.math.squared_difference(fake_disc_A,1))
        disc_loss_B = tf.reduce_mean(tf.math.squared_difference(fake_disc_B,1))
        
        gen_loss_A = cyclic_loss*10 + disc_loss_B
        gen_loss_B = cyclic_loss*10 + disc_loss_A
        
        d_loss_A = (tf.reduce_mean(tf.square(fake_pool_disc_A)) + tf.reduce_mean(tf.math.squared_difference(real_disc_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(fake_pool_disc_B)) + tf.reduce_mean(tf.math.squared_difference(real_disc_B,1)))/2.0

        return gen_loss_A,d_loss_A,gen_loss_B,d_loss_B
    
    
    def model_opti(self,gen_loss_A,disc_loss_A,gen_loss_B,disc_loss_B,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()
        discA_vars = [var for var in train_vars if var.name.startswith('CycleGAN/discriminator_A')]
        discB_vars = [var for var in train_vars if var.name.startswith('CycleGAN/discriminator_B')]
        genA_vars = [var for var in train_vars if var.name.startswith('CycleGAN/generator_A')]        
        genB_vars = [var for var in train_vars if var.name.startswith('CycleGAN/generator_B')]        
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            discA_train_opt = tf.compat.v1.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_A,var_list = discA_vars)
            discB_train_opt = tf.compat.v1.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_B,var_list = discB_vars)
            genA_train_opt = tf.compat.v1.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_A,var_list = genA_vars)
            genB_train_opt = tf.compat.v1.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_B,var_list = genB_vars)
    
        return discA_train_opt, discB_train_opt, genA_train_opt, genB_train_opt
        
    
def Fetch_New_Image(image_path):
    image = Image.open(image_path)
    image = image.resize([128,128],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)
    return image
    
        
        
        
        
