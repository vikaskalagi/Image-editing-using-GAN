import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()
import numpy as np 
from glob import glob
import scipy.misc
import os 
import random
import shutil 

from cycleGAN_model import *

def test(cgan_net, season1, season2, model_dir, images_dir):

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        print(os.getcwd())
        model_dir = "./"+model_dir+"/"
        print("{}.data-00000-of-00001".format(model_dir))

        if os.path.exists("{}.data-00000-of-00001".format(model_dir)):
            saver = tf.train.Saver()
            saver.restore(session, model_dir)
            print("Model Restore successful")

        else:        
            print("Model Path not present....recheck")
            return
        
        image_A = np.reshape(Fetch_New_Image(season1), (1, 128, 128, 3))
        image_B = np.reshape(Fetch_New_Image(season2), (1, 128, 128, 3))
        
        images = session.run([cgan_net.gen_A, cgan_net.gen_B], feed_dict={cgan_net.input_A:image_A, cgan_net.input_B:image_B})
        
        img_ind = 1 

        for image in images:
            image = np.reshape(image, (128, 128, 3))
            image = (((image - image.min())*255)/(image.max()-image.min())).astype(np.uint8)
            scipy.misc.toimage(image).save(images_dir+"/season_change_"+str(img_ind)+".jpg")
            img_ind = img_ind + 1
            

def cycle_gan_test_wrapper(_):

    dps_path = "summer2winter_yosemite"
    model_dir = "cycle_gan_model_save/try_200"
    images_dir = "cycle_gan_testing_op"

    list_files = glob("./"+images_dir+"/*.jpg")
    for files in list_files:
        os.remove(files)

    summer_image = random.choice(glob("./" + dps_path + "/testA/*.jpg"))
    winter_image = random.choice(glob("./" + dps_path + "/testB/*.jpg"))
    
    shutil.copyfile(summer_image, "./" + images_dir + "/original_1.jpg")
    shutil.copyfile(winter_image, "./" + images_dir + "/original_2.jpg")

    print(summer_image)
    print(winter_image)

    input_shape = (128, 128, 3)
    pool_size = 50 
    beta1 = 0.5
    tf.reset_default_graph()
    
    cgan_net = CycleGAN(input_shape, pool_size, beta1)
    test(cgan_net, summer_image, winter_image, model_dir, images_dir)                       


tf.app.run(cycle_gan_test_wrapper)
