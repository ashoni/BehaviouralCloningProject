import pickle
import numpy as np
import math
import csv
import cv2
from sklearn.utils import shuffle
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import warnings
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')


#Load initial data
data = pd.read_csv('driving_log.csv', header = None)
data.columns = ["center_images","left_images","right_images","steering","brake","throttle","speed"]
labels = [float(x) for x in data['steering'][1:]]
images = data['center_images'][1:]
left_images = data['left_images'][1:]
right_images = data['right_images'][1:]

left_labels = [x + 0.22 for x in labels]
right_labels = [x - 0.22 for x in labels]

print('Data loaded.')


#Flip all images to positive angle and merge them to one list
def flipAndSave(imagepath, toFlip):
    imagepath = imagepath.replace(' ', '')
    image = cv2.imread(imagepath, 1)
    if toFlip:
        image = cv2.flip(image, 1)
    flippedPath = imagepath.replace('IMG', 'FLIPPED')
    cv2.imwrite(flippedPath, image)
    return flippedPath


n_images = []
n_labels = []
for l, im in zip(labels, images):
    flipped = flipAndSave(im, l < 0)
    n_images.append(flipped)
    n_labels.append(abs(l))

for l, im in zip(left_labels, left_images):
    flipped = flipAndSave(im, l < 0)
    n_images.append(flipped)
    n_labels.append(abs(l))

for l, im in zip(right_labels, right_images):
    flipped = flipAndSave(im, l < 0)
    n_images.append(flipped)
    n_labels.append(abs(l))

# Creating images classes.
elems = {}
for i in range(len(n_labels)):
    r = int(round(n_labels[i], 1) * 10)
    if r in elems:
        elems[r] += [i]
    else:
        elems[r] = [i]

class_weights = {0: 5518,
                 1: 3482,
                 2: 11033,
                 3: 1999,
                 4: 1272,
                 5: 474,
                 6: 196,
                 7: 96,
                 8: 16,
                 9: 11,
                 10: 0,
                 11: 0,
                 12: 0}

# Image processing functions
img_width = 64
img_height = 64

def shift_image(image, angle):
    angle_per_px = 0.003
    y_size = image.shape[0]
    x_size = image.shape[1]
    shift = np.random.randint(-60, 60)
    t_angle = angle + shift * angle_per_px
    if t_angle > 1:
        t_angle = 1
    if t_angle < -1:
        t_angle = -1
    transformation = np.float32([[1, 0, shift], [0, 1, 0]])
    t_image = cv2.warpAffine(image, transformation, (x_size, y_size))
    return t_image, t_angle

def add_shadow(image):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            region_select = np.copy(image)
            region_select = cv2.cvtColor(region_select, cv2.COLOR_BGR2HSV)
            coeff_bright = np.random.uniform(0.5, 0.8)
            region_select[:, :, 2] = region_select[:, :, 2] * coeff_bright
            region_select = cv2.cvtColor(region_select, cv2.COLOR_HSV2BGR)

            y_size = image.shape[0]
            x_size = image.shape[1]
            border_bottom = [random.choice(range(x_size)), y_size]
            border_top = [random.choice(range(x_size)), 0]
            shadow_line = np.polyfit((border_bottom[0], border_top[0]), (border_bottom[1], border_top[1]), 1)
            XX, YY = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))
            if np.random.uniform() < 0.5:
                region_thresholds = (YY > (XX * shadow_line[0] + shadow_line[1]))
            else:
                region_thresholds = (YY < (XX * shadow_line[0] + shadow_line[1]))
            image[region_thresholds] = region_select[region_thresholds]
            return image
        except (np.RankWarning, RuntimeWarning):
            return image

def loadAndPreprocessImage(imagepath, label):
    imagepath = imagepath.replace(' ', '')
    imagepath = imagepath.replace('IMG', 'FLIPPED')
    image = cv2.imread(imagepath, 1)

    if np.random.uniform() > 0.5:
        image = add_shadow(image)

    if np.random.uniform() > 0.5:
        image = cv2.flip(image, 1)
        label = -label

    image, label = shift_image(image, label)

    # random brightness
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    coeff_bright = 0.6 + np.random.uniform() * 0.6
    image[:, :, 2] = image[:, :, 2] * coeff_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # crop
    rows = image.shape[0]
    cols = image.shape[1]
    image = image[math.floor(6 * rows / 15):rows - 23, 0:cols]

    # resize
    image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)

    return image, label

def batchGenerator(images, labels, counters, weights, batch_size=32):
    batch_images = np.zeros((batch_size, img_height, img_width, 3),dtype=np.uint8)
    batch_labels = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            part = random.choice(weights)
            item = random.choice(counters[part])
            batch_images[i], batch_labels[i] = loadAndPreprocessImage(images[item], labels[item])
        yield batch_images, batch_labels

def batchMean(b_images, b_labels, b_counters, b_weights):
    for b_set_x, b_set_y in batchGenerator(b_images, b_labels, b_counters, b_weights, 512):
        break
    return np.mean([i ** 2 for i in b_set_y])




warnings.filterwarnings("ignore", category=DeprecationWarning)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(img_width, img_height, 3)))

model.add(Convolution2D(3, 1, 1, input_shape=(img_width, img_height, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, init='normal', activation='elu'))
model.add(Dense(64, init='normal', activation='elu'))
model.add(Dense(16, init='normal', activation='elu'))
model.add(Dense(1, init='normal', activation='linear'))


t_images, t_labels = shuffle(n_images, n_labels)
t_images, v_images = np.split(t_images, [int(0.8 * len(t_images))])
t_labels, v_labels = np.split(t_labels, [int(0.8 * len(t_labels))])

t_elems = {}
for i in range(len(t_labels)):
    r = int(round(t_labels[i], 1) * 10)
    if r in t_elems:
        t_elems[r] += [i]
    else:
        t_elems[r] = [i]

t_weights = []
for key in t_elems.keys():
    t_weights += [key] * class_weights[key]

print("train stat")
for key in t_elems.keys():
    print(key, ' ', len(t_elems[key]))
print("train mean = ", batchMean(t_images, t_labels, t_elems, t_weights))

v_elems = {}
for i in range(len(v_labels)):
    r = int(round(v_labels[i], 1) * 10)
    if r in v_elems:
        v_elems[r] += [i]
    else:
        v_elems[r] = [i]

v_weights = []
for key in v_elems.keys():
    v_weights += [key] * class_weights[key]

print("val stat")
for key in v_elems:
    print(key, ' ', len(v_elems[key]))
print("val mean = ", batchMean(v_images, v_labels, v_elems, v_weights))

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit_generator(batchGenerator(t_images, t_labels, t_elems, t_weights, 64),
                              samples_per_epoch=64 * 96 * 3,
                              nb_epoch=10,
                              validation_data=batchGenerator(v_images, v_labels, v_elems, v_weights, batch_size=64),
                              nb_val_samples=64 * 24 * 3)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('model.h5')




