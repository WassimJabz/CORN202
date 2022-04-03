import numpy as np
import os
import tensorflow as tf
import pandas as pd
import collections 
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D
from keras.models import Sequential
from keras import regularizers
from keras.initializers import RandomNormal
from skimage import data, img_as_float
from skimage import exposure


path = os.getcwd()
DATASET_PATH = os.path.join(path,"dataset\\")


lb_to_str_csv = DATASET_PATH + 'label_int_to_str_mapping.csv'

lb_to_str_df = pd.read_csv(lb_to_str_csv)

TEST_NPY = DATASET_PATH + 'test_images.npy'
test_images_us = np.load(TEST_NPY)

def contrast_stretch(img_set_us):
    img_set = []
    for img_us in img_set_us:
        img_us = np.reshape(img_us, (28,28)) / 255
        p2, p98 = np.percentile(img_us, (2, 98))
        img =  exposure.rescale_intensity(img_us, in_range=(p2, p98))
        img_set.append(img)
    return img_set

test_images = contrast_stretch(test_images_us)

test_images = np.reshape(test_images, (-1, 28, 28,1))
kFoldPred = []
finalPred = []
for i in range(1,6):
    model = tf.keras.models.load_model("saved_model/Model 4_contrast/fold"+str(i))

    y_test = model.predict(test_images)
    y_test = np.argmax(y_test, axis=1)
    kFoldPred.append(y_test)

for i in range(len(kFoldPred[0])):
    co = collections.Counter([row[i] for row in kFoldPred])
    co = sorted(co.items(),key=lambda x: x[1],reverse=True)
    finalPred.append(co[0][0])

df_test = pd.read_csv(os.path.join(path, 'dataset\\sample_submission.csv'))
df_test['label'] = finalPred
df_test.to_csv(os.path.join(path, 'dataset\\submission.csv'), index=False)

df_test2 = pd.read_csv(os.path.join(path, 'dataset\\sample_submission.csv'))
for i in range(5):
    df_test2['label'+str(i)] = kFoldPred[i]
df_test2.to_csv(os.path.join(path, 'dataset\\kFolds.csv'), index=False)