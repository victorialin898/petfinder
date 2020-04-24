import numpy as np
import tensorflow as tf
from scipy import signal, io, interpolate
import os
import random
from PIL import Image

import hyperparameters as hp
import csv
import shutil


# A script to set up the train and test datasets to work with flow_from_directory
# Make sure you have a data/ file in the root directory, and the full dataset from kaggle 
#  is inside this firectory and called "petfinder-adoption-prediction"
def create_sets(data_path, is_train=True):

    # WHY ARE THERE COMMAS IN THE COMMA SEPARATED VALUE FILE FML

    if is_train:
        info_path = "petfinder-adoption-prediction/train/train.csv"
        dest_path = os.path.join(data_path, "train")
        imgs_path = os.path.join(data_path, "petfinder-adoption-prediction/train_images/")
    else:
        # Test data does not have labels...
        pass

    id_dict = {}
    print("Building pet ID to adoption speed dictionary.")

    with open(os.path.join(data_path, info_path)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # TRAIN: Row -1 is adoption speed, -2 is PhotoAmt, -3 is PetID
                id_dict[row[-3]] = row[-1]
                line_count += 1
        print(f'Processed {line_count} lines.')

    try:
        os.mkdir(dest_path)
    except OSError as e:
        print ("\tCreation of the directory %s failed: %s" % (dest_path, e.strerror))
    else:
        print ("\tSuccessfully created the directory %s " % dest_path)

    print("Making and populating class folders. ")
    for i in range(hp.category_num):
        path = os.path.join(dest_path, str(i))
        try:
            os.mkdir(path)
        except OSError as e:
            print ("\tCreation of the directory %s failed: %s" % (path, e.strerror))
        else:
            print ("\tSuccessfully created the directory %s " % path)

    succ = 0
    fails = []
    for root, _, files in os.walk(imgs_path):

        for name in files:
            if name.endswith(".jpg"):
                img_path = os.path.join(root, name)
                if name.split("-")[0] in id_dict:
                    shutil.copyfile(img_path, os.path.join(*[dest_path, str(id_dict[name.split("-")[0]]), name]))
                    succ += 1
                else:
                    fails.append(name)
    print("Successfully copied %d images, failed to copy %d" %(succ, len(fails)))



# Preprocess code
class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.category_num

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()


        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=(0,1,2))
        self.std = np.std(data_sample, axis=(0,1,2))

        # ==========================================================

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
        return (img-self.mean)/self.std

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        img = img / 255.
        img = self.standardize(img)
        return img


    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
            # Documentation for ImageDataGenerator: https://bit.ly/2wN2EmK
            #
            # ============================================================

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                preprocessing_function=self.preprocess_fn)

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        img_size = hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
