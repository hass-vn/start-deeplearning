import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
import keras


DATADIR = "/home/hass/Desktop/cat-and-dog/training_set"
CATEGORIER = ["dogs", "cats"]
training_data = []

def creat_training_data():
    for category in CATEGORIER:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIER.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), 0)
                new_array = cv2.resize(img_array, (64, 64))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


creat_training_data()
training = training_data[:8000]
random.shuffle(training)

X = []
Y = []

for feature, lable  in training:
    X.append(feature)
    Y.append(lable)

x_train = np.array(X).reshape(-1, 64, 64, 1)
y_train = np.array(Y)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
x_train = x_train/255.0

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(16))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=40 )
model.save('my_model_v2.h5')