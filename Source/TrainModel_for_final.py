import tensorflow as tf
import tensorflow.contrib.slim as slim
from load_data_t import load_dataset
from sklearn.model_selection import train_test_split
import time
import os
import numpy as np
import math
#place holder [none,  h, w, c]
model_path = './model/saved_final/'
keep_drop = 0.5
batch_size = 550
epoch = 10000
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2" # set gpu Device


def model(x, keep_drop=keep_drop):
    x_norm = tf.cast(x,tf.float32) / 255.0 #data normalize

    net = slim.conv2d(x_norm , 64, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 64 64

    net = slim.conv2d(net, 128, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 32 32

    net = slim.conv2d(net, 256, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 16 16

    net = slim.conv2d(net, 512,kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 8 8

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))
    net = slim.max_pool2d(net, (2, 2)) # 4 4

    net = slim.conv2d(net, 1024, kernel_size=(3, 3))
    net = tf.nn.dropout(net, keep_prob=0.6)
    net = slim.max_pool2d(net, (2, 2)) # 2 2

    net = tf.nn.dropout(net, keep_prob=0.6)
    net = slim.conv2d(net, 1024, kernel_size=(2, 1), padding='VALID') # 1 2 1024

    net = tf.nn.dropout(net, keep_prob=0.5)
    net = slim.conv2d(net, 4, kernel_size=(1, 1)) # 1 2 4

    net = tf.squeeze(net, axis =1) # 2 4

    return net

def cal_max_distance(pred, y): #calculate maximum distance / 1 batch_size
    max_dis = 0
    for i in range(len(pred)):
        max = -1
        a = pred[i].reshape(4, 2)
        b = y[i].reshape(4, 2)
        for j in range(4):
            if max < np.abs(b[j][0]-a[j][0]):
                max = np.abs(b[j][0]-a[j][0])
            if max < np.abs(b[j][1]-a[j][1]):
                max = np.abs(b[j][1]-a[j][1])
        max_dis += max / len(pred)
    return max_dis

if __name__ == "__main__":
    images, labels, _, w_h = load_dataset(True)
    images, test_images, labels, test_labels, w_h_train, w_h_test = train_test_split(images, labels, w_h, train_size=0.8, random_state=42)

    print('Verify Data...')
    print("Data's shape: ",images.shape, labels.shape, w_h_train.shape)
    print(np.max(images), np.min(images),np.std(images), np.mean(images))
    print(np.max(labels), np.min(labels), np.std(labels), np.mean(labels))

    print('First Label: ', labels[0])
    print('First W/H', w_h[0])
    #exit()

    X = tf.placeholder(tf.uint8, [None, images.shape[1], images.shape[2], images.shape[3]], name='input')
    Y = tf.placeholder(tf.float32, [None, 2, 4], name='Y')

    logits = model(X, keep_drop)
   # exit()
    with tf.name_scope('Optimizer'):
        cost = tf.reduce_mean(tf.abs(Y-logits)) # MAE
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08).minimize(cost)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model/saved_final') # for model restore
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored.')
        else:
            sess.run(tf.global_variables_initializer())
        print('Learning Start')

        start_sec = time.time()

        for ep in range(epoch):
            f = open("./model/saved_final/log.txt", 'a')
            fy_list_test = []
            fy_list = []
            this_time = time.time()
            save_time = this_time - start_sec > 60
            now = time.strftime("%H:%M:%S", time.localtime())
            avg_cost = 0

            avg_mean_dis = 0
            avg_mean_dis_test = 0
            start_sec = time.time()
            total_batch = int(len(images) / batch_size)
            for i in range(total_batch): # Training
                start = i * batch_size
                end = (i+1) *  batch_size
                batch_x = images[start:end]
                batch_y = labels[start:end]
              #  batch_wh = w_h_train[start:end]

                feed_dict = {X: batch_x, Y: batch_y}
                _, _y_hat, c = sess.run([optimizer, logits, cost], feed_dict=feed_dict)
                avg_mean_dis += cal_max_distance(_y_hat, batch_y) / total_batch
                avg_cost += np.sqrt(c) / total_batch
                fy_list.extend(_y_hat)


            avg_cost_test = 0
            avg_mean_dis_test = 0
            total_batch_test = int(len(test_images) / batch_size)



            for i in range(total_batch_test): # Test
                start = i * batch_size
                end = (i+1) *  batch_size
                batch_x = test_images[start:end]
                batch_y = test_labels[start:end]
                feed_dict_test = {X:batch_x, Y: batch_y}
                _y_hat_test, cost_t = sess.run([logits, cost], feed_dict=feed_dict_test)
                avg_cost_test += np.sqrt(cost_t) / total_batch_test
                avg_mean_dis_test += cal_max_distance(_y_hat_test, batch_y) / total_batch_test
                fy_list_test.extend(_y_hat_test)

            f.write('epoch:'+str(ep)+" Train "+str(avg_cost)+" "+str(avg_mean_dis)+" "+str(avg_cost_test)+" "+str(avg_mean_dis_test)+"\n");


            if ep % 10 == 0:
                print("Epoch: %04d (train cost: %.5f dis: %.5f) (test cost: %.5f dis: %.5f)" % (ep+1 ,avg_cost,avg_mean_dis, avg_cost_test, avg_mean_dis_test))

            if ep%50 == 0:
                print(fy_list[0])
                print(labels[0])

            if ep % 100 == 0:
                start_sec = this_time
                save_path = saver.save(sess, os.path.join(model_path, 'regression_model.ckpt'))
                print("Model Saved, time: %s, %s",(now, save_path))
            f.close()
