# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 18:13
# @Author  : sunjiawei
# @FileName: b.py

# 模型定义

import os
import time
import datetime

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

# from layers_test import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer

IMGSIZE = 112
N_LANDMARK = 68


def NormRmse(GroudTruth, Prediction):
    Gt = tf.reshape(GroudTruth, [-1, N_LANDMARK, 2])
    Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)   ###  求每个样本 68 个关键点的预测误差
    # loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((Gt - Pt)**2,axis=1)))
    # norm = tf.sqrt(tf.reduce_sum(((tf.reduce_mean(Gt[:, 36:42, :],1) - \
    #     tf.reduce_mean(Gt[:, 42:48, :],1))**2), 1))
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :], 1) - tf.reduce_mean(Gt[:, 42:48, :], 1), axis=1)
    return loss / norm

def DAN(MeanShapeNumpy):  ## 初始输入：
    MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32,name='MeanShape')
    InputImage = tf.placeholder(tf.float32, [None, IMGSIZE, IMGSIZE, 1],name='InputImage')
    GroundTruth = tf.placeholder(tf.float32, [None, N_LANDMARK * 2],name='GroundTruth')
    S1_isTrain = tf.placeholder(tf.bool,name='S1_isTrain')
    S2_isTrain = tf.placeholder(tf.bool,name='S2_isTrain')
    Ret_dict = {}
    Ret_dict['InputImage'] = InputImage
    Ret_dict['GroundTruth'] = GroundTruth
    Ret_dict['S1_isTrain'] = S1_isTrain
    Ret_dict['S2_isTrain'] = S2_isTrain

    with tf.variable_scope('Stage1'):
        S1_Conv1a = tf.layers.batch_normalization(
            tf.layers.conv2d(InputImage, 64, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Conv1b = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Conv1a, 64, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b, 2, 2, padding='same')

        S1_Conv2a = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Pool1, 128, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Conv2b = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Conv2a, 128, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b, 2, 2, padding='same')

        S1_Conv3a = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Pool2, 256, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Conv3b = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Conv3a, 256, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b, 2, 2, padding='same')

        S1_Conv4a = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Pool3, 512, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Conv4b = tf.layers.batch_normalization(
            tf.layers.conv2d(S1_Conv4a, 512, 3, 1, padding='same', activation=tf.nn.relu,
                             kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b, 2, 2, padding='same')

        S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4)
        S1_DropOut = tf.layers.dropout(S1_Pool4_Flat, 0.5, training=S1_isTrain)

        S1_Fc1 = tf.layers.batch_normalization(
            tf.layers.dense(S1_DropOut, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer()),
            training=S1_isTrain, name='S1_Fc1')
        S1_Fc2 = tf.layers.dense(S1_Fc1, N_LANDMARK * 2, name='S1_Fc2')

        S1_Ret = tf.add(S1_Fc2, MeanShape, name="S1_Ret")
        S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, S1_Ret), name="S1_Cost")  ### S1_Ret其实是相当于S1

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "Stage1"))

    Ret_dict['S1_Ret'] = S1_Ret
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Optimizer'] = S1_Optimizer

    return Ret_dict