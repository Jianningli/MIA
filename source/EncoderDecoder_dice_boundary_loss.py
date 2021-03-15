from __future__ import print_function, division
import os
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import UpSampling3D, Conv3D
from tensorflow.keras.layers import Conv3DTranspose as Deconv3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import hamming_loss
from utils import mkdirs
from glob import glob
import random
import nrrd
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt as distance


''' Codes adapted from https://github.com/LIVIAETS/boundary-loss.
    Credit goes to the original authors
'''




def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def surface_loss_keras(y_true, y_pred):
    y_true = y_true.numpy()
    gt_dist_transform=np.array([calc_dist_map(y) for y in y_true]).astype(np.float32) 
    multipled = y_pred * gt_dist_transform
    return K.mean(multipled)



def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_boundary_loss(y_true,y_pred):
    loss1=surface_loss_keras(y_true,y_pred)
    loss2=dice_coef_loss(y_true,y_pred)
    finalloss=loss1 + 100*loss2
    return finalloss


def build_generator():

    model = Sequential()

    model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=vol_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv3D(128, kernel_size=5, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv3D(512, kernel_size=1, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    model.add(UpSampling3D())
    model.add(Deconv3D(256, kernel_size=5, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Deconv3D(128, kernel_size=5, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling3D())
    model.add(Deconv3D(64, kernel_size=5, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling3D())
    model.add(Deconv3D(channels, kernel_size=5, padding="same"))
    model.add(Activation('tanh'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling3D())
    model.add(Deconv3D(channels, kernel_size=5, padding="same"))
    model.add(Activation('sigmoid'))


    model.summary()


    return model



def resizing(label):
    a,b,c=label.shape
    resized_data = zoom(label,(128/a,128/b,64/c),order=2, mode='constant')  
    return resized_data

def resizing_up(label):
    resized_data = zoom(label,(4,4,2),order=2, mode='constant')  
    return resized_data


def save_model(MODEL_DIR):
    def save(model, model_name):
        model_path = os.path.join(MODEL_DIR, "%s.h5" % model_name)
        model.save(model_path)
    save(generator, "dice_boundary_loss")



def train(generator,MODEL_DIR, epochs, batch_size=16, sample_interval=50):


    ipt=np.load('defective_skull.npy')
    gt=np.load('gt_implant.npy')

    for epoch in range(epochs):
        print(epoch)
        idx = np.random.randint(0, ipt.shape[0], batch_size)
        masked_vols=ipt[idx]
        missing_parts=gt[idx]
      
        
        print('masked_vols:',masked_vols.shape)
        print('missing_parts:',missing_parts.shape)

        g_loss = generator.train_on_batch(masked_vols, missing_parts)
        print(g_loss)
        if epoch % sample_interval == 0:
            save_model(MODEL_DIR)




def evaluate(testdir,test_results_dir):
    print('evaluating the model...')

    test_list=glob('{}/*.nrrd'.format(testdir))
    for i in range(len(test_list)):
        data,h=nrrd.read(test_list[i])
        data=data[:,:,data.shape[2]-128:data.shape[2]]
        data=resizing(data)
        data=np.expand_dims(np.expand_dims(data,axis=0),axis=4)
        gen_missing = generator.predict(data)

        gen_missing=(gen_missing>0)
        gen_missing=gen_missing+1-1
        gen_missing_up=resizing_up(gen_missing[0,:,:,:,0])
        filename1=test_results_dir+test_list[i][-10:-5]+'.nrrd'
        nrrd.write(filename1,gen_missing[0,:,:,:,0],h)
        filename2=test_results_dir+'resized/'+test_list[i][-10:-5]+'.nrrd'
        nrrd.write(filename2,gen_missing_up,h)



if __name__ == '__main__':
    print(tf.executing_eagerly())
    vol_rows = 128
    vol_cols = 128
    vol_height = 64
    mask_height = 128
    mask_width = 128
    mask_length = 64
    channels = 1
    num_classes = 2
    vol_shape = (vol_rows, vol_cols, vol_height, channels)
    missing_shape = (mask_height, mask_width, mask_length, channels)
    MODEL_DIR = '../dice_boundary_loss'
    generator = build_generator()
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss=dice_boundary_loss,optimizer=optimizer,run_eagerly=True)
    masked_vol = Input(shape=vol_shape)
    train(generator,MODEL_DIR,epochs=3000, batch_size=4, sample_interval=200)
