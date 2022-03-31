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

model = tf.keras.models.load_model("saved_model/my_model")

path = os.getcwd()
DATASET_PATH = os.path.join(path,"dataset\\")


lb_to_str_csv = DATASET_PATH + 'label_int_to_str_mapping.csv'

lb_to_str_df = pd.read_csv(lb_to_str_csv)

TEST_NPY = DATASET_PATH + 'test_images.npy'
test_images = np.load(TEST_NPY)
test_images = np.reshape(test_images, (-1, 28, 28,1))/255

y_test = model.predict(test_images)
y_test = np.argmax(y_test, axis=1)

df_test = pd.read_csv(os.path.join(path, 'dataset\\sample_submission.csv'))
df_test['label'] = y_test
df_test.to_csv(os.path.join(path, 'dataset\\submission.csv'), index=False)