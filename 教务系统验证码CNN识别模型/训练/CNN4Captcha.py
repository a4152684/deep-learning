
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from helpers import resize_to_fit


LETTER_IMAGES_FOLDER = "labeled_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER): #paths将该文件夹下的文件名变成列表
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#颜色空间转换函数,将BGR格式转换成灰度图片

    # Resize the letter so it fits in a 30x30 pixel box
    image = resize_to_fit(image, 30, 30)

    # Add a third channel dimension to the image to make Keras happy#这个happy也太秀了
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2] #os.path.sep路径分隔符，一般是"/"

    # Add the letter image and it's label to our training data
    data.append(image)
    #print(data)
    labels.append(label)
    #print(labels)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)    #转换为numpy的格式
    
# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
#print(X_train.shape)
#print(Y_train)
# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)#标签二值化，将标签转换为一个[0,1]矩阵，其中指标的顺序由决策树来决定
#lb用inverse_transform转换回来
Y_train = lb.transform(Y_train) #解码就是lb.inverse_transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)#序列化,并将结果数据流写入到文件对象中，load是解码

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(30, 30, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# Hidden layer with 500 nodes
model.add(Flatten())#展平成一维
model.add(Dense(500, activation="relu"))

# Output layer with 36 nodes (one for each possible letter/number we predict)
model.add(Dense(36, activation="softmax"))
model.build()
model.summary()
# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])#优化

# Train the neural network
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
model.fit(X_train, Y_train, epochs=10)

model.evaluate(X_test, Y_test)

#model.save_weights('model_weights.h5')

# Save the trained model to disk
model.save(MODEL_FILENAME)

