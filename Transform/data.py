from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import gc
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from scipy.misc import imsave

class dataProcess(object):
   

    def __init__(self, out_rows, out_cols, data_path = "./MVC_image_pairs_resize_new/shirts_1", mask_path = "./MVC_image_pairs_resize_new/fc8_mask_5_modified", label_path = "./MVC_image_pairs_resize_new/shirts_5", test_data_path = "./test/shirts_1", test_mask_path = "./test/fc8_mask_5_modified", npy_path = "./npydata", img_type = "jpg", mask_type = ""):

     
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.mask_path = mask_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_data_path = test_data_path
        self.test_mask_path = test_mask_path
        self.npy_path = npy_path
        self.resize_size = 256

    def create_train_data(self):
        i = 0
        
        print('-'*30)
        print('Creating training images...')
        print('-'*30)
        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        print(len(imgs))
        
        imgdatas = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)
        imgmasks = np.ndarray((len(imgs), self.resize_size, self.resize_size, 1), dtype=np.float32)
        imglabels = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)

        if not os.path.exists(self.npy_path):
            os.mkdir(self.npy_path)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            maskname = midname.split("_")[0] + "_5.jpg"

            img = load_img(self.data_path + "/" + midname, grayscale = False)
            img_mask = load_img(self.mask_path + "/" + maskname, grayscale = True)
            label = load_img(self.label_path + "/" + maskname, grayscale = False)

            img = img.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img_mask = img_mask.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            label = label.resize( (self.resize_size, self.resize_size), Image.BILINEAR )


            img = img_to_array(img)
            img_mask = img_to_array(img_mask)
            label = img_to_array(label)
 
            imgdatas[i] = img
            imgmasks[i] = img_mask
            imglabels[i] = label

            
            
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
          
            i += 1
        
        return imgdatas, imgmasks, imglabels


    def create_test_data(self):
        i = 0

        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = glob.glob(self.test_data_path+"/*."+self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)
        imgmasks = np.ndarray((len(imgs), self.resize_size, self.resize_size, 1), dtype=np.float32)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            maskname = midname.split("_")[0] + "_5.jpg"
            img = load_img(self.test_data_path + "/" + midname, grayscale = False)
            img = img.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img = img_to_array(img)
           
            img_mask = load_img(self.test_mask_path + "/" + maskname, grayscale = True)
            img_mask = img_mask.resize( (self.resize_size, self.resize_size), Image.BILINEAR )
            img_mask = img_to_array(img_mask)
            imgdatas[i] = img
            imgmasks[i] = img_mask
            i += 1
        print('loading done')
       
        return imgdatas, imgmasks

if __name__ == "__main__":
    mydata = dataProcess(641, 641)
    mydata.create_train_data()
    mydata.create_test_data()
 
