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
def create_sets(data_path, train_ratio):

    # WHY ARE THERE COMMAS IN THE COMMA SEPARATED VALUE FILE FML

    info_path = "petfinder-adoption-prediction/train/train.csv"
    dest_path_train = os.path.join(data_path, "train")
    dest_path_test = os.path.join(data_path, "test")
    imgs_path = os.path.join(data_path, "petfinder-adoption-prediction/train_images/")

    try:
        os.mkdir(dest_path_train)
        os.mkdir(dest_path_test)
    except OSError as e:
        print ("\tCreation of the directories %s, %s failed: %s" % (dest_path_train, dest_path_test, e.strerror))
        return
    else:
        print ("\tSuccessfully created the directories %s, %s " % (dest_path_train, dest_path_test,))

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


    print("Making and populating class folders. ")
    for i in range(hp.category_num):
        path_train = os.path.join(dest_path_train, str(i))
        path_test = os.path.join(dest_path_test, str(i))
        try:
            os.mkdir(path_train)
            os.mkdir(path_test)
        except OSError as e:
            print ("\tCreation of the directories %s, %s failed: %s" % (path_train, path_test, e.strerror))
        else:
            print ("\tSuccessfully created the directories %s, %s " % (path_train, path_test))

    train_amt = 0
    test_amt = 0
    fails = []
    for root, _, files in os.walk(imgs_path):

        for name in files:
            if name.endswith(".jpg"):
                img_path = os.path.join(root, name)
                if name.split("-")[0] in id_dict:
                    if (train_amt / hp.img_count < train_ratio):
                        shutil.copyfile(img_path, os.path.join(*[dest_path_train, str(id_dict[name.split("-")[0]]), name]))
                        train_amt += 1
                    else:
                        shutil.copyfile(img_path, os.path.join(*[dest_path_test, str(id_dict[name.split("-")[0]]), name]))
                        test_amt += 1
                else:
                    fails.append(name)
    print("Successfully copied %d images to train, %d to test, %d total. Failed to copy %d" %(train_amt, test_amt, train_amt + test_amt, len(fails)))



# Datasets object that holds the test/train data and information
class Datasets():
    """ Class that holds train/data sets and information, structured 
    much like the Datasets() class in project 4
    """

    def __init__(self, data_path):
        self.data_path = data_path

        self.idx_to_class = {}
        self.class_to_idx = {}
        self.classes = [""] * hp.category_num

        # For standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()

        # Data generators
        self.train_data = self.get_data(os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(os.path.join(self.data_path, "test/"), False, False)

    def calc_mean_and_std(self):
        """ Takes sample of dataset and calculates a sample. I used a lot
        of my project 4 code since it was very relevant here!
        
        Arguments: none
        Returns: none
        """

        # Walk through unzipped Petfinder Data folder, retrieve and shufflefile names
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))
        random.shuffle(file_list)

        # Retrieve sample based on prespecified hyperparameter
        file_list = file_list[:hp.preprocess_sample_size]

        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        self.mean = np.mean(data_sample, axis=(0,1,2))
        self.std = np.std(data_sample, axis=(0,1,2))

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))


    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator- an interesting addition 
        if we have time could be to implement different preprocessing methods """

        img = img / 255.
        img = (img-self.mean)/self.std
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
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                preprocessing_function=self.preprocess_fn)

        else:
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        img_size = hp.img_size
        classes_for_flow = None

        # Align data generators by label
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=None)

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
