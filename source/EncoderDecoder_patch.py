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


IMAGE_DIR = './32_cube/images'
MODEL_DIR = './32_cube/saved_model/AutoEncoder_patch'



''' Codes adapted from https://github.com/Fdevmsy/3D_shape_inpainting.
    Credit goes to the original authors
'''


class EncoderDecoder():
    def __init__(self):
        self.vol_rows = 128
        self.vol_cols = 128
        self.vol_height = 128
        self.mask_height = 128
        self.mask_width = 128
        self.mask_length = 128
        self.channels = 1
        self.num_classes = 2
        self.vol_shape = (self.vol_rows, self.vol_cols, self.vol_height, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.mask_length, self.channels)

        self.input_dir   = "../defective_skull_train"
        self.gt_imp_dir   = "../gt_implants_train"


        optimizer = Adam(0.0002, 0.5)

        try:
            #self.discriminator = load_model(os.path.join(MODEL_DIR, 'discriminator.h5'))
            self.generator = load_model(os.path.join(MODEL_DIR, 'encoderdecoder_patch.h5'))

            print("Loaded checkpoints")
        except:
            self.generator = self.build_generator()
            #self.discriminator = self.build_discriminator()
            print("No checkpoints found")

        # discriminator
        #self.discriminator.compile(loss='binary_crossentropy',
        #                           optimizer=optimizer,
        #                           metrics=['accuracy'])

        # generator
        # The generator takes noise as input and generates the missing part
        masked_vol = Input(shape=self.vol_shape)

        gen_missing = self.generator(masked_vol)

        # For the combined model we will only train the generator
        #self.discriminator.trainable = False

        # The discriminator takes generated voxels as input and determines
        # if it is generated or if it is a real voxels
        #valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_vol, gen_missing)
        self.combined.compile(loss='mse',
                              #loss=['mse', 'binary_crossentropy'],
                              #loss_weights=[0.9, 0.1],
                              optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        # Encoder
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.vol_shape, padding="same"))
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
        model.add(Deconv3D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation('tanh'))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling3D())
        model.add(Deconv3D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation('tanh'))


        model.summary()

        masked_vol = Input(shape=self.vol_shape)
        gen_missing = model(masked_vol)

        return Model(masked_vol, gen_missing)


    def train(self, epochs, batch_size=16, sample_interval=50):

        #X_train = self.generateWall()
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print('loading data...')
        #(shape=(85,16,1,128,128,128,1))
        ipt=np.load('ipt_patch.npy')
        gt=np.load('gt_imp_patch.npy')
        print('loading data complete...')


        for epoch in range(epochs):
            idx=random.randrange(0,85,1)
            #masked_vols, missing_parts, _ = self.mask_randomly(vols)
            masked_vols=ipt[idx]
            #nrrd.write('masked_vols.nrrd',masked_vols[0,:,:,:,0],h)
            missing_parts=gt[idx]
            #nrrd.write('missing_parts.nrrd',missing_parts[0,:,:,:,0],h)
            #masked_vols: (5, 32, 32, 32, 1)
            print('masked_vols:',masked_vols.shape)
            #missing_parts: (5, 16, 16, 16, 1)
            print('missing_parts:',missing_parts.shape)
            for i in range(16):
                # Train Generator
                g_loss = self.combined.train_on_batch(masked_vols[i], missing_parts[i])
                print(g_loss)
            print('epochs:',epoch)
            # save generated samples
            if epoch % sample_interval == 0:
                #idx = np.random.randint(0, X_train.shape[0], 2)
                #vols = X_train[idx]
                #self.sample_images(epoch, vols)
                self.save_model()

    def make_patch(self,label):
        label_list=[]
        for x in range(4):
            for y in range(4):
                temp_label=np.expand_dims(np.expand_dims(label[x*128:(x+1)*128,y*128:(y+1)*128,:],axis=0),axis=4)
                label_list.append(temp_label)
        return np.array(label_list)


    def evaluate(self, testdir,test_results_dir):
        print('evaluating the model...')

        test_list=glob('{}/*.nrrd'.format(testdir))
        for i in range(len(test_list)):
            data,h=nrrd.read(test_list[i])
            data=data[:,:,data.shape[2]-128:data.shape[2]]
            datap=self.make_patch(data)

            reconstructed=np.zeros(shape=(512,512,128))
            patch_idx=0
            for x in range(4):
                for y in range(4):
                    gen_missing = self.generator.predict(datap[patch_idx])
                    gen_missing=(gen_missing>0.5)
                    gen_missing=gen_missing+1-1
                    reconstructed[x*128:(x+1)*128,y*128:(y+1)*128,:]=gen_missing[0,:,:,:,0]
                    patch_idx=patch_idx+1            
            filename=test_results_dir+test_list[i][-10:-5]+'.nrrd'
            nrrd.write(filename,reconstructed,h)


    def save_model(self):
        def save(model, model_name):
            model_path = os.path.join(MODEL_DIR, "%s.h5" % model_name)
            model.save(model_path)

        save(self.generator, "encoderdecoder_patch")
        #save(self.discriminator, "discriminator")


if __name__ == '__main__':
    test_dir="../defective_skull_test"
    test_results_dir="../results/"
    context_encoder = EncoderDecoder()
    context_encoder.train(epochs=3000, batch_size=4, sample_interval=200)
    #context_encoder.evaluate(test_dir,test_results_dir)


