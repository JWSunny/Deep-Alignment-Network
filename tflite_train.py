# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 18:13
# @Author  : sunjiawei
# @FileName: a.py

##训练部分的代码
import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'

import tensorflow as tf
import numpy as np

from ImageServer import ImageServer
from tflite_replay import DAN
import cv2

datasetDir = "../data/result/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=192540_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=200_perturbations=[]_size=[112, 112].npz")

# Create a new DAN regressor.
# 两个stage训练，不宜使用 estimator
# regressor = tf.estimator.Estimator(model_fn=DAN, params={})

## 每个数据样本的标签---其实是68个关键点的坐标位置
def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))
nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]
Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
# import pdb; pdb.set_trace()
Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)
testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))
meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))
S1_isTrain = tf.placeholder(tf.bool)
S2_isTrain = tf.placeholder(tf.bool)
dan = DAN(initLandmarks)
STAGE = 1
with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)
    if STAGE < 2:
        sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(sess, './A/DAN-Menpo.ckpt')
        print('Pre-trained model has been loaded!')
    print("Starting training......")
    for epoch in range(150):
        print(epoch)
        Count = 0
        while Count * 128 < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0], 10, False)
            sess.run(dan['S1_Optimizer'],feed_dict={dan['InputImage']: Xtrain[RandomIdx], dan['GroundTruth']: Ytrain[RandomIdx], dan['S1_isTrain']: True,dan['S2_isTrain']: False})
            Count += 1
        tf.train.write_graph(sess.graph_def, "./A/", "DAN_Menpo_graph.pb", as_text=False)
        Saver.save(sess, './A/DAN-Menpo.ckpt')
