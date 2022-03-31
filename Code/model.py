import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt

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
