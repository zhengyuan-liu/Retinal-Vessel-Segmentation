# ==========================================================
#
#  This prepare the hdf5 datasets of the CHASE database
#
# ============================================================

import os
import h5py
import numpy as np
from PIL import Image


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
# train
CHASE_DIR = './CHASEDB1/Fold_1/'
original_imgs_train = CHASE_DIR + "Training/Images/"
groundTruth_imgs_train = CHASE_DIR + "Training/Labels/"
borderMasks_imgs_train = CHASE_DIR + "Training/Masks/"
# test
IMG_TEST_DIR = CHASE_DIR + "Testing/Images/"
groundTruth_imgs_test = CHASE_DIR + "Testing/Labels/"
MASK_TEST_DIR = CHASE_DIR + "Testing/Masks/"
# ---------------------------------------------------------------------------------------------

channels = 3
width = 999
height = 960
# dataset_path = "./CHASE_datasets_training_testing/"
dataset_path = "./CHASE_datasets_training_testing/"


def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, Nimgs):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        i = 0
        for file in files:
            if not file.endswith('.jpg'):
                continue
            # original
            print("original image: " + file)
            img = Image.open(imgs_dir + file)
            imgs[i] = np.asarray(img)
            # corresponding ground truth
            groundTruth_name = file[0:9] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name).convert('L')
            groundTruth[i] = np.asarray(g_truth)
            # corresponding border masks
            border_masks_name = "mask_" + file[6:9] + ".png"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            i += 1

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    print("groundTruth max: " + str(np.max(groundTruth)))
    print("groundTruth min: " + str(np.min(groundTruth)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
# getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train, 21)
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "CHASE_dataset_img_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "CHASE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + "CHASE_dataset_mask_train.hdf5")

# getting the testing datasets
img_test, groundTruth_test, mask_test = get_datasets(IMG_TEST_DIR, groundTruth_imgs_test, MASK_TEST_DIR, 7)
print("saving test datasets")
write_hdf5(img_test, dataset_path + "CHASE_dataset_img_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "CHASE_dataset_groundTruth_test.hdf5")
write_hdf5(mask_test, dataset_path + "CHASE_dataset_mask_test.hdf5")
