# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:13:55 2021

@author: Deepak Malekar
"""

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)
import matplotlib.pyplot as plt

# Variables
imdir = "D:/computer-vision/basedata/test"
targetdir = "D:/cluster"
number_clusters = 2

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.jpg'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())
#Importing required modules
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np 
#Load Data
data = load_digits().data
pca = PCA(2)
#Transform the data
df = pca.fit_transform(data)
#Import KMeans module
from sklearn.cluster import KMeans
#Initialize the class object
kmeans = KMeans(n_clusters= 2)
#predict the labels of clusters.
label = kmeans.fit_predict(df)
#Getting unique labels
u_labels = np.unique(label) 
#plotting the result#s:
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()
