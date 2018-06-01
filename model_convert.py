# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 11:13
# @Author  : sunjiawei
# @FileName: model_convert.py

from argparse import ArgumentParser

import tensorflow as tf

from tensorflow.python.framework import graph_util
from ImageServer import ImageServer

# A workaround to fix an import issue
# see https://github.com/tensorflow/tensorflow/issues/15410#issuecomment-352189481

IMGSIZE = 112
N_LANDMARK = 68


def DAN(MeanShapeNumpy):
    # Create a model to classify single image
    MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32, name='MeanShape')
    InputImage = tf.placeholder(tf.float32, [None, IMGSIZE, IMGSIZE, 1], name='InputImage')
    GroundTruth = tf.placeholder(tf.float32, [None, N_LANDMARK * 2], name='GroundTruth')
    S1_isTrain = tf.placeholder(tf.bool, name='S1_isTrain')
    S2_isTrain = tf.placeholder(tf.bool, name='S2_isTrain')
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
        # S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, S1_Ret), name="S1_Cost")  ### S1_Ret其实是相当于S1

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Stage1')):
        #     S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost, var_list=tf.get_collection(
        #         tf.GraphKeys.TRAINABLE_VARIABLES, "Stage1"))

    Ret_dict['S1_Ret'] = S1_Ret

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state( './DAN_Menpo_Model_test/')
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring %s" % ckpt.model_checkpoint_path)
            # Restore the model trained by train.py
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph_def = tf.get_default_graph().as_graph_def()
            # Freeze the graph
            output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ["Stage1/S1_Ret"])
            # The input type and shape of the converted model is inferred from the input_tensors argument
            tflite_model = tf.contrib.lite.toco_convert(output_graph, input_tensors=[InputImage], output_tensors=[S1_Ret])
            with open('./DAN_Menpo_Model_test/DAN-Menpo.tflite', "wb") as f:
                f.write(tflite_model)
        else:
            print("Checkpoint not found")

if __name__ == '__main__':
    datasetDir = "../data/result/"
    trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=192540_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
    validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=200_perturbations=[]_size=[112, 112].npz")

    initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))

    DAN(initLandmarks)

    print("converting ok!")