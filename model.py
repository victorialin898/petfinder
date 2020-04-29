import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import hyperparameters as hp

def cnn():

    model = Sequential([
        # Block 1
        Conv2D(64, 3, padding="same", activation="relu", name="block1_conv1", input_shape=(hp.img_size,hp.img_size,3)), # im assuming we have color images?
        Conv2D(64, 3, padding="same", activation="relu", name="block1_conv2"),
        MaxPool2D(2, name="block1_pool"),
        # Block 2
        Conv2D(128, 3, padding="same", activation="relu", name="block2_conv1"),
        Conv2D(128, 3, padding="same", activation="relu", name="block2_conv2"),
        MaxPool2D(2, name="block2_pool"),
        # Block 3
        Conv2D(256, 3, padding="same", activation="relu", name="block3_conv1"),
        Conv2D(256, 3, padding="same", activation="relu", name="block3_conv2"),
        Conv2D(256, 3, padding="same", activation="relu", name="block3_conv3"),
        MaxPool2D(2, name="block3_pool"),
        # Block 4
        Conv2D(512, 3, padding="same", activation="relu", name="block4_conv1"),
        Conv2D(512, 3, padding="same", activation="relu", name="block4_conv2"),
        Conv2D(512, 3, padding="same", activation="relu", name="block4_conv3"),
        MaxPool2D(2, name="block4_pool"),
        # Block 5
        Conv2D(512, 3, padding="same", activation="relu", name="block5_conv1"),
        Conv2D(512, 3, padding="same", activation="relu", name="block5_conv2"),
        Conv2D(512, 3, padding="same", activation="relu", name="block5_conv3"),
        MaxPool2D(2, name="block5_pool"),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(hp.category_num, activation="softmax")
    ])
    model.summary()
    return model
