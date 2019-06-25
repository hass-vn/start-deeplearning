import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model



model = load_model('my_model_v2.h5')
img_ = cv2.imread('t10.jpg', 0)
img = cv2.resize(img_, (64, 64))


img = np.array(img)
img = img.reshape(-1, 64, 64, 1)
img = img/255.0

pred_img = model.predict(img)
pred = np.argmax(pred_img)
pred = int(pred)
plt.imshow(img_)
if pred == 1:
    plt.xlabel('con_meo')
else:
    plt.xlabel('con_cho')

plt.show()