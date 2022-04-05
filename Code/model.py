import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adamax, Adam
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import regularizers
from keras.initializers import RandomNormal
import seaborn as sns
from sklearn.model_selection import StratifiedKFold , KFold ,RepeatedKFold
from sklearn.model_selection import train_test_split
from skimage import data, img_as_float
from skimage import exposure
from skimage import data, img_as_float
from skimage import exposure


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



show_img = np.reshape(test_images[38], (28,28))  

train_images = np.reshape(train_images, (-1, 28,28,1)) /255

test_images = np.reshape(test_images, (-1, 28, 28,1)) /255



class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            print("reached threshold, stopping training")
            self.model.stop_training = True



train_size = 50000

labels = train_lb_df['label'].tolist()
x_train = train_images[:train_size]
y_train = np.array(labels)[:train_size]

#x_val = train_images[train_size:]
#y_val = np.array(labels)[train_size:]



#y_val = to_categorical(y_val)
kFold = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

fold_no = 1

acc_per_fold = []
loss_per_fold = []

for train, test in kFold.split(x_train,y_train):
    fold_y_train = y_train[train]
    fold_y_train = to_categorical(fold_y_train)

    fold_y_test = y_train[test]
    fold_y_test = to_categorical(fold_y_test)


    model = Sequential()

    model.add(Conv2D(32, 3, strides=1, padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    #model.add(MaxPooling2D(2,2))

    model.add(Conv2D(32, 3, strides=1,padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.30))


    #model.add(MaxPooling2D(2,2))

    model.add(Conv2D(32, 3, strides=1,padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))
    model.add(Conv2D(64, 3, strides=1,padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.30))
    model.add(Flatten())
    model.add(Dense(350, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.60))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))




    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    my_callback = MyThresholdCallback(threshold=0.9100)

    history = model.fit(x_train[train], fold_y_train, epochs=180, batch_size=256, validation_data=(x_train[test], fold_y_test),callbacks=[my_callback])

    model.save(f'saved_model/my_model/fold{fold_no}')

    scores = model.evaluate(x_train[test], fold_y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

    #history_df = pd.DataFrame(history.history)
    #fig = plt.figure(figsize=(15,4), facecolor="#97BACB")
    #fig.suptitle("Learning Plot of Model for Loss")
    #pl=sns.lineplot(data=history_df["accuracy"],color="#444160")
    #pl=sns.lineplot(data=history_df["val_accuracy"],color="#146160")
    #pl.set(ylabel ="Accuracy")
    #pl.set(xlabel ="Epochs")
    #plt.show()

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')








