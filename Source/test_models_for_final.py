import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from load_data_t import load_dataset
import numpy as np
import math
from sklearn.model_selection import train_test_split
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
keep_drop = 0.5
def model(x, keep_drop=keep_drop):
    x_norm = tf.cast(x,tf.float32) / 255.0 #data normalize

    net = slim.conv2d(x_norm , 64, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 32

    net = slim.conv2d(net, 128, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 32
    print(net)
    net = slim.conv2d(net, 256, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 32
    print(net)
    net = slim.conv2d(net, 512,kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 16 16

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2))# 8 8

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))
    net = tf.nn.dropout(net, keep_prob=1)
    net = slim.max_pool2d(net, (2, 2)) # 4 4

    net = tf.nn.dropout(net, keep_prob=1)
    net = slim.conv2d(net, 1024, kernel_size=(2, 1), padding='VALID')
    net = tf.nn.dropout(net, keep_prob=1)
    net = slim.conv2d(net, 4, kernel_size=(1, 1))

    net = tf.squeeze(net, axis =1)
  #  net = tf.contrib.layers.batch_norm(net)

    return net


def cal_max_error(pred, y):
    lis = []
    pred = pred.reshape(4, 2)
    y = y.reshape(4, 2)
    max = -1
    for j in range(4):
        if max < np.abs(y[j][0]-pred[j][0]):
            max = np.abs(y[j][0]-pred[j][0])
        if max < np.abs(y[j][1]-pred[j][1]):
            max = np.abs(y[j][1]-pred[j][1])

    return max

if __name__ =="__main__":

    images, labels, paths, w_h = load_dataset()
    _, images, _, labels, _, paths, _, w_h = train_test_split(images, labels, paths, w_h, train_size = 0.8, random_state=42)
    print(images)

    X = tf.placeholder(tf.uint8, [None, images.shape[1], images.shape[2], images.shape[3]])
    Y = tf.placeholder(tf.float32, [None, 2, 4], name='Y')

    logits = model(X, 1)
    acc = 0
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model/saved_final')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored.')
        else:
            sess.run(tf.global_variables_initializer())

        import matplotlib.pyplot as plt
        for i in range(len(images)):
           # print(images[i:i+1].size)

            feed_dict1 = {X: images[i:i+1]}
            #print(type(feed_dict1))
            _y_hat = sess.run(logits, feed_dict = feed_dict1)
            # print(_y_hat)
            # print(labels[i])
            # print(np.array(_y_hat[0] * 128).astype(np.int))
            vertex = np.array(_y_hat[0] * 128).astype(np.int)

            if cal_max_error(_y_hat[0], labels[i]) < 0.1:
                acc +=  1 / len(images)
            # #else: # Compare Lagel with Prediction
            #
            #     images[i][vertex[0][1]:vertex[0][1]+2, vertex[0][0]:vertex[0][0]+2] = (255, 0, 0)
            #     images[i][vertex[0][3]:vertex[0][3]+2, vertex[0][2]:vertex[0][2]+2] = (255, 0, 0)
            #     images[i][vertex[1][1]:vertex[1][1]+2, vertex[1][0]:vertex[1][0]+2] = (255, 0, 0)
            #     images[i][vertex[1][3]:vertex[1][3]+2, vertex[1][2]:vertex[1][2]+2] = (255, 0, 0)
            #     ver_lab = np.array(labels[i]*128).astype(np.int)
            #     images[i][ver_lab[0][1]:ver_lab[0][1]+2, ver_lab[0][0]:ver_lab[0][0]+2] = (0, 255, 0)
            #     images[i][ver_lab[0][3]:ver_lab[0][3]+2, ver_lab[0][2]:ver_lab[0][2]+2] = (0, 255, 0)
            #     images[i][ver_lab[1][1]:ver_lab[1][1]+2, ver_lab[1][0]:ver_lab[1][0]+2] = (0, 255, 0)
            #     images[i][ver_lab[1][3]:ver_lab[1][3]+2, ver_lab[1][2]:ver_lab[1][2]+2] = (0, 255, 0)
            #
            #
            #
            #     plt.imshow(images[i])
            #     plt.title("Max error: %.3f"%(cal_max_error(_y_hat[0], labels[i])))
            #     plt.show()
    print(acc)
