# 此代码用来验证我们的模型结果

import os
import pickle

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from helpers import resize_to_fit
from sklearn.preprocessing import LabelBinarizer
from JpgPretreat import cfs, clearNoise, saveImage, saveSmall, twoValue

test_images_file = 'test_images'
extracted_letter_images = 'temps'
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

if not os.path.exists(extracted_letter_images):
    os.makedirs(extracted_letter_images)

n = 0
for file in os.listdir(test_images_file):
    image = Image.open(test_images_file + '/' + file).convert('L')
    twoValue(image, 100)
    clearNoise(image, 1, 1)
    image = saveImage(image.size)
    x, y = cfs(image)
    n += 1
    saveSmall(extracted_letter_images, str(n) + '.jpg', image, x, y)

data = []
for image_file in os.listdir(extracted_letter_images):
    # Load the image and convert it to grayscale
    image = cv2.imread(extracted_letter_images + '/' + image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 30, 30)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)
    data.append(image)

model = tf.keras.models.load_model(MODEL_FILENAME)
# lb = LabelBinarizer().fit(data)
# data = lb.transform(data)
data = np.array(data, dtype=np.float32)
predicted = model.predict(data)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

labels = lb.inverse_transform(predicted)
print("预测结果为:")
for i in range(int(len(labels) / 4)):
    print(labels[4 * i] + labels[4 * i + 1] + labels[4 * i + 2] +
          labels[4 * i + 3])
