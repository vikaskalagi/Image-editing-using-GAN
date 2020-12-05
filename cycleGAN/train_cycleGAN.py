import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()
import numpy as np 
from glob import glob
import random
from PIL import Image 
import os
import scipy.misc

from cycleGAN_model import *


def save_to_pool(poolA, poolB, gen_A, gen_B, pool_size, num_im):
        
        if num_im < pool_size:
            poolA[num_im] = gen_A
            poolB[num_im] = gen_B
        
        else:
            p = random.random()
            if p > 0.5:
                indA = random.randint(0,pool_size-1)
                poolA[indA] = gen_A
            p = random.random()
            if p > 0.5: 
                indB = random.randint(0,pool_size-1)
                poolB[indB] = gen_B
        
        num_im = num_im + 1
        return poolA,poolB,num_im


def train(cycle_gan_network,max_img,trainA,trainB,lr_rate,shape,pool_size,model_dir,images_dir):
    saver = tf.train.Saver(max_to_keep=None)
    lenA = len(trainA)
    lenB = len(trainB)
    epoch = 0
    summ_count = 0 
    wint_count = 0
    num_imgs = 0
    poolA = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))
    poolB = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        #saver.restore(session,model_dir+"try_60\\")
        while epoch < 201:
            if epoch >= 100:
                lr_rate = 0.0002 - ((epoch-100)*0.0002)/100
            
            for step in range(max_img):
                
                if summ_count >= lenA:
                    summ_count = 0
                    random.shuffle(trainA)
                
                if wint_count >= lenB:
                    wint_count = 0
                    random.shuffle(trainB)
                
                  
                summer_image = Fetch_New_Image(trainA[summ_count])
                winter_image = Fetch_New_Image(trainB[wint_count])

                summ_count = summ_count + 1
                wint_count = wint_count + 1
                
                summer_image = np.reshape(summer_image,(1,shape[0],shape[1],shape[2]))
                winter_image = np.reshape(winter_image,(1,shape[0],shape[1],shape[2]))
               
                
                _,genB,genA_loss,_,genA,genB_loss,cyclicA,cyclicB = session.run([cycle_gan_network.genA_opt,cycle_gan_network.gen_B,cycle_gan_network.gen_loss_A,cycle_gan_network.genB_opt,cycle_gan_network.gen_A,cycle_gan_network.gen_loss_B,cycle_gan_network.cyclicA,cycle_gan_network.cyclicB], feed_dict={cycle_gan_network.input_A:summer_image,cycle_gan_network.input_B:winter_image,cycle_gan_network.lr_rate:lr_rate})
                
                poolA, poolB, num_imgs = save_to_pool(poolA, poolB, genA, genB, pool_size, num_imgs)
                
                indA = random.randint(0,(min(pool_size,num_imgs)-1))
                indB = random.randint(0,(min(pool_size,num_imgs)-1))
                fakeA_img = poolA[indA]
                fakeB_img = poolB[indB]
                
                
                _,discA_loss,_,discB_loss = session.run([cycle_gan_network.discA_opt,cycle_gan_network.disc_loss_A,cycle_gan_network.discB_opt,cycle_gan_network.disc_loss_B], feed_dict={cycle_gan_network.input_A:summer_image,cycle_gan_network.input_B:winter_image,cycle_gan_network.lr_rate:lr_rate,cycle_gan_network.fake_pool_Aimg:fakeA_img,cycle_gan_network.fake_pool_Bimg:fakeB_img})
                
                if step % 50 == 0:
                    print ("epoch = %r step = %r discA_loss = %r genA_loss = %r discB_loss = %r genB_loss = %r" %(epoch,step,discA_loss,genA_loss,discB_loss,genB_loss))
                    
                if step % 150 == 0:
                    images = [genA, cyclicB, genB, cyclicA]
                    img_ind = 0
                    for img in images:
                        img = np.reshape(img,(shape[0],shape[1],shape[2]))
                        if np.array_equal(img.max(),img.min()) == False:
                            img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
                        else:
                            img = ((img - img.min())*255).astype(np.uint8)
                        scipy.misc.toimage(img, cmin=0.0, cmax=...).save(images_dir+"/img_"+str(img_ind)+"_"+str(epoch)+"_"+str(step)+".jpg")
                        img_ind = img_ind + 1
                        
                print ("step = %r" %(step))
                
            if epoch % 50 == 0:
                saver.save(session, model_dir + "/try_"+str(epoch)+"/", write_meta_graph=True)
            
            epoch = epoch + 1
        
        
def cycle_gan_train_wrapper(_): 
    
    dps_path = "summer2winter_yosemite"
    model_dir = "cycle_gan_model_save"
    trained_images_save = "cycle_gan_training_op"
    
    trainA = glob("./" + dps_path + "/trainA/*.jpg")
    trainB = glob("./" + dps_path + "/trainB/*.jpg")
    
    print(trainA)
    print(trainB)
    print("\n\n")

    input_shape = (128, 128, 3)
    max_img = 8
    pool_size = 50 
    lr_rate = 0.0002
    beta1 = 0.5
    
    tf.reset_default_graph()
    
    cycle_gan_network = CycleGAN(input_shape, pool_size, beta1)
    
    train(cycle_gan_network, max_img, trainA, trainB, lr_rate, input_shape, pool_size, model_dir, trained_images_save)

    
tf.app.run(cycle_gan_train_wrapper)
