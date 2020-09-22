###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import argparse

from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, merge
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
# from keras.utils.visualize_util import plot
# from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from help_functions import *

# function to obtain data for training/testing (validation)
from extract_patches import get_data_training_rotate


# U-Net
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)
    #
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model


parser = argparse.ArgumentParser(description='Code Coverage')
parser.add_argument('--dataset', '-d', action='store', default='DRIVE', help='Dataset name, default: DRIVE')
parser.add_argument('--config', '-c', action='store', default='../configuration_drive.txt',
                    help='Path of Configuration file, default: ../configuration_drive.txt')
parser.add_argument('--pretained_model', '-p', action='store', default='../configuration_drive.txt',
                    help='Path of Pretained model file, default: ../configuration_drive.txt')
args = parser.parse_args()

config_name = args.config

# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('./' + config_name)
# patch to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
experiment_name = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

datasets = {'DRIVE', 'STARE', 'CHASE'}
dataset_name = args.dataset_name
if dataset_name not in datasets:
    print("Dataset NOT support!")
    exit(1)
print("Dataset: ", dataset_name)


# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training_rotate(
    train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV'),  # select the patches only inside the FOV  (default == True)
    dataset=dataset_name
)


# ========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5), '../' + experiment_name + '/' + "sample_input_imgs")
visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5), '../' + experiment_name + '/' + "sample_input_masks")


# =========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
# model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
# print("Check: final output of the network:")
# print(model.output_shape)
# plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
# json_string = model.to_json()
# open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)


# Load the saved model
experiment_name = config.get('training settings', 'pretain_model')
path_pretained_model = './' + experiment_name + '/'
model = model_from_json(open(path_pretained_model + experiment_name + '_architecture.json').read())
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
best_last = config.get('training settings', 'best_last')
# model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
model.load_weights(path_pretained_model + experiment_name + '_' + best_last + '_weights.h5')
print("Successfully load pretained model!")


# ============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./' + experiment_name + '/' + experiment_name + '_best_weights.h5', verbose=1,
                               monitor='val_loss', mode='auto', save_best_only=True)  # save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True,
          validation_split=0.1, callbacks=[checkpointer])


# ========== Save and test the last model ===================
model.save_weights('./' + experiment_name + '/' + experiment_name + '_last_weights.h5', overwrite=True)
# test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

