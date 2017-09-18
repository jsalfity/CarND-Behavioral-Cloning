import itertools
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from scipy import misc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import cv2
import math
#import json
import matplotlib.pyplot as plt


from sklearn.utils import shuffle


def load_data():
    csv_path = './data_mouse_windy2/driving_log.csv'
    # Get center, left, and right pics
    center_pics = np.genfromtxt(csv_path, delimiter=',', usecols=(0,), unpack=True, dtype=str,skip_header = 1)

    left_pics = np.genfromtxt(csv_path, delimiter=',', usecols=(1,), unpack=True, dtype=str, skip_header=1)

    right_pics = np.genfromtxt(csv_path, delimiter=',', usecols=(2,), unpack=True, dtype=str, skip_header=1)

    steering_angle = np.genfromtxt(csv_path, delimiter=',', usecols=(3,), unpack=True, dtype=str,
                                   skip_header=1)

    X_center = center_pics
    X_left = left_pics
    X_right = right_pics
    y_train = steering_angle

    X_center, X_left, X_right, y_train = shuffle(X_center, X_left, X_right, y_train)

    y_train = y_train.astype(np.float)

    center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val = train_test_split(
         X_center,
         X_left,
         X_right,
         y_train,
         test_size=0.10,
         random_state=832289)

    return center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val

def NVIDIA_model():
 
     # input shape=64x64x3
    model = Sequential()

    # Lambda Layer
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(64, 64, 3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    #model.add(Dropout(0.50))

    # Add two 3x3 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    
    model.add(Dropout(0.50))

    #add flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    #output layer
    model.add(Dense(1))

    model.summary()
    adam_op = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=adam_op, metrics=['mean_absolute_error'])

    return model


## PROCESSING IMAGES
def crop_resize(image, image_size):
    #Resize image to 100x100 and crop top and bottom 20% to make interpretation easier for model
    shape = image.shape
    # image = image[int(shape[0] * 0.2):int(shape[0] * 0.80), 0:shape[1]]
    image = image[int(shape[0]*0.40):int(shape[0]*0.85), 0:shape[1]]
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image

def shift_horiz_vert(image, steering_angle, range_x, range_y):
    # Shift image horizontally/vertically, while adjusting steering angle by +/- 0.002
    # shift_x = range_adj * np.random.uniform() - range_adj / 2
    # shift_y = 40 * np.random.uniform() - 40 / 2
    # steer_ang = steer + shift_x / range * 2 * .2
    # trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # image_tr = cv2.warpAffine(image, trans, (image_size, image_size))
    # return image_tr, steer_ang
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    # Apply random brightness to image
    bright_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bright_image = cv2.cvtColor(bright_image,cv2.COLOR_RGB2HSV)
    bright_image[:,:,2] = bright_image[:,:,2]*(.25+np.random.uniform())
    bright_image= cv2.cvtColor(bright_image,cv2.COLOR_HSV2RGB)
    return bright_image

def flip(image, angle):
    # Randomly flips image and negates steering angle
    if np.random.randint(2) == 0:
        image = cv2.flip(image, 1)
        angle = - angle
    return image, angle

def get_generator(train_features, left_features, right_features, train_labels, threshold, batch_size, image_size):
        # Yield batch at a time to fit_generator using random data. Data will be chosen randomly between center, left, and right
        # images, and steering angle and images will be augmented
        images = np.zeros((batch_size, image_size, image_size, 3))
        angles = np.zeros(batch_size)
        while 1:
            for i in range(batch_size):
                # Generate random index value from batch
                rand_line = np.random.randint(len(train_labels))
                # Get a batch of training features and labels. 
                image, label = get_rand(train_features[rand_line], left_features[rand_line], right_features[rand_line], train_labels[rand_line], image_size)
                images[i] = image
                angles[i] = label
            yield images, angles

def get_rand(center_pic, left_pic, right_pic, angle, image_size):
    # Randomly chooses between center, left, or right picture and adjust steering angle
    rand_pic = np.random.randint(3)
    if (rand_pic == 0):
        path = left_pic.strip()
        angle += 0.25
    if (rand_pic == 1):
        path = center_pic.strip()
    if (rand_pic == 2):
        path = right_pic.strip()
        angle -= 0.25
    path = center_pic.strip()

    # Apply brightness and image shifting augmentation techniques
    image = cv2.imread(path)
    image = random_brightness(image)
    image = crop_resize(image, image_size)
    image, angle = flip(image, angle)
    image, angle = shift_horiz_vert(image, angle, 100, 10)
    return image, angle

def main():
    center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val = load_data()

    model = NVIDIA_model()

    threshold = 1
    batch_size = 256
    n_train_epochs = 5
    image_size = 64

    for i_pr in range(n_train_epochs):
        # Obtain training and validation generators and use fit_generator to test data. Incrementally reduce threshold to eliminate
        # extreme values
        train_r_generator = get_generator(center_train, left_train, right_train, y_train, threshold, batch_size, image_size)
        val_r_generator = get_generator(center_train, left_train, right_train, y_train, threshold, batch_size, image_size)
        model.fit_generator(train_r_generator, samples_per_epoch=50000, nb_epoch=1, validation_data=val_r_generator, nb_val_samples= 10,
                            verbose=1)
        #threshold = 1/(i_pr + 1)*1
        print("FINIHED EPOCH ", (i_pr + 1), "of" ,n_train_epochs)
    # json_string = model.to_json()
    # with open('model.json', 'w') as outfile:
    #      json.dump(json_string, outfile)

    model.save("modelNVIDIA_windy7.h5")
    print("model saved")
    pass


if __name__ == '__main__':
    main()