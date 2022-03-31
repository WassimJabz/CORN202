import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D
from keras.models import Sequential
from keras import regularizers
from keras.initializers import RandomNormal



path = os.getcwd()


DATASET_PATH = os.path.join(path,"dataset\\")


lb_to_str_csv = DATASET_PATH + 'label_int_to_str_mapping.csv'
train_lb_csv = DATASET_PATH + 'train_labels.csv'

lb_to_str_df = pd.read_csv(lb_to_str_csv)
lb_to_str_df.head()
train_lb_df = pd.read_csv(train_lb_csv)
train_lb_df.head()

TRAIN_NPY = DATASET_PATH + 'train_images.npy'
TEST_NPY = DATASET_PATH + 'test_images.npy'
train_images = np.load(TRAIN_NPY)
test_images = np.load(TEST_NPY)
# print(train_images[0])
show_img = np.reshape(train_images[5], (28,28))
train_images = np.reshape(train_images, (-1, 28,28,1)) / 255
test_images = np.reshape(test_images, (-1, 28, 28,1)) / 255
plt.imshow(show_img)
plt.show()



train_size = 40000

labels = train_lb_df['label'].tolist()
x_train = train_images[:train_size]
y_train = np.array(labels)[:train_size]

x_val = train_images[train_size:]
y_val = np.array(labels)[train_size:]


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


model = Sequential()

model.add(Conv2D(8, 3, strides=1, activation='relu', input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.0625, seed=None)))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val))


