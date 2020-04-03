 # -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:37:01 2020

@author: True C
"""
import cv2
import numpy as np
import pickle
#from imutils import paths
from PIL import Image
import tensorflow as tf
from helpers import resize_to_fit
from JpgPretreat import img2list

CAPTCHA_IMAGINE_FOLDER = "captcha_images"
MODEL_FILENAME = "../训练/captcha_model.hdf5"
MODEL_LABELS_FILENAME = "../训练/model_labels.dat"

model =tf.keras.models.load_model(MODEL_FILENAME)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)#标签解码

result_list=[]
#for image_file in paths.list_images(CAPTCHA_IMAGINE_FOLDER):
for i in range(20):
    image_file=CAPTCHA_IMAGINE_FOLDER+"/"+str(i)+".jpg"
    image = Image.open(image_file).convert("L")
    image_list= img2list(image)#分割到4个长度的列表内
    for j in range(len(image_list)):
        image_list[j]= image_list[j].convert("RGB")
        image_list[j]= cv2.cvtColor(np.asarray(image_list[j]),cv2.COLOR_RGB2GRAY)#转换为opencv的格式
        image_list[j] = resize_to_fit(image_list[j], 30, 30) #转换到30x30
        image_list[j] = np.expand_dims(image_list[j], axis=2) #扩展 
    image_list = np.array(image_list, dtype="float") / 255.0
    
    predicted_score_matrix = model.predict(image_list)#预测，输出跟标签二值化后的Y_train类似
    label=lb.inverse_transform(predicted_score_matrix)#LabelBinarizer的方法
    result_list.append(label.tolist())
    print(label.tolist())

#print(result_list)

    

