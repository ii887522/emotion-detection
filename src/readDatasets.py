import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

Datadirectory = "train/"
Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

## read all images and convert them to array
training_Data = []
img_size = 224

for category in Classes:
    path = os.path.join(Datadirectory, category)
    class_num = Classes.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_Data.append([new_array,class_num])
        except Exception as e:
            pass

## shuffle so the algorithm does not follow the sequence
random.shuffle(training_Data)

x = [] ## feature, data
y = [] ## label of data (expressions)

for features, label in training_Data:
    x.append(features)
    y.append(label)

X = np.array(x).reshape(-1, img_size, img_size, 3) ## convert into 4 dimension

## normalize the data before training
X = X/255.0;