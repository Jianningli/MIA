from __future__ import print_function, division

import os
import numpy as np
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D, Deconv3D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import hamming_loss
from utils import mkdirs
from glob import glob
import random
import nrrd
from scipy.ndimage import zoom
from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance







def surface_loss_keras(y_true, y_pred):
    multipled = y_pred * y_true
    return K.mean(multipled)



def build_generator():

    model = Sequential()

    # Encoder
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

    # Decoder
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
    model.add(Activation('tanh'))


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
    save(generator, "boundaryloss")


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def train(generator,MODEL_DIR, epochs, batch_size=16, sample_interval=50):


    ipt=np.load('ipt_85_128_128_64.npy')
    gt=np.load('gt_denoised.npy')

    for epoch in range(epochs):
        print(epoch)
        idx = np.random.randint(0, ipt.shape[0], batch_size)
        masked_vols=ipt[idx]
        missing_parts=gt[idx]
        
        gt_dist_transform=np.array([calc_dist_map(y) for y in missing_parts]).astype(np.float32)
        
        print('masked_vols:',masked_vols.shape)
        print('missing_parts:',missing_parts.shape)
        print('gt_dist_transform:',gt_dist_transform.shape)
        # Train Generator
        g_loss = generator.train_on_batch(masked_vols, gt_dist_transform)
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
    test_dir="../defective_skull_test"
    test_results_dir="../results_ae_boundary/"
    MODEL_DIR = '../boundarylosss'
    mkdirs(MODEL_DIR)
    try:
        generator = load_model('../boundaryloss.h5',custom_objects={'surface_loss_keras': surface_loss_keras})
        print("Loaded checkpoints")
    except:
        generator = build_generator()
        print("No checkpoints found")

    masked_vol = Input(shape=vol_shape)
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss=surface_loss_keras,optimizer=optimizer)
    train(generator,MODEL_DIR,epochs=3000, batch_size=4, sample_interval=200)
    #evaluate(test_dir,test_results_dir)


