# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 14:16
# @Author  : sunjiawei
# @FileName: ImageLandmarks_test.py

## 主要对脚本 TrainingSetPreparation 产生的数据（所有图片集，best关键点，文件提供的关键点）；
## 通过代码实现图片集中关于关键点的标记；以确保是人脸的关键点

import tensorflow as tf
import numpy as np
from ImageServer import ImageServer
from models_test import DAN
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
##  加载数据
N_LANDMARK = 68
datasetDir = "../data/result/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")

##  获取每个数据样本的标签---其实是68个关键点的坐标位置
def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]
    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks
    return y.reshape((nSamples, nLandmarks * 2))
## 获取样本数，样本的尺寸及通道
nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]
##  获取图片数据及对应图片中关键点的位置
Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)


##  在样本图片中绘制出关键点的位置；验证是否是人脸的关键点
##  cv2.circle中坐标必须是int类型
for i in range(len(Xtrain)):
    img=Xtrain[i]
    img=np.reshape(img,[112,112])
    # initialize=np.zeros((112,112),dtype='float32')
    points=[int(round(Ytrain[i][j])) for j in range(len(Ytrain[i]))] ## float转int类型
    points=np.reshape(points,(N_LANDMARK,2))
    for j in range(points.shape[0]):
        pos = (points[j, 0], points[j, 1])
        cv2.circle(img, pos, 0, color=(255, 0, 0))   ## color=(255, 0, 0)代表红色
        plt.imshow(img)
        cv2.waitKey(0)
    plt.close()
