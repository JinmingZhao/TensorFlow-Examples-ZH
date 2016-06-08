#coding: utf-8
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
'''
dropout = 0.75
iters = 2000  accuracy = 0.332031
iters = 20000  accuracy = 0.921875

dropout = 0.5
iters  =20000, accuracy = 0.914062
'''
training_iters = 4000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28),转化为一维的输入
n_classes = 10 # MNIST total classes (0-9 digits)
# Dropout, probability to keep units, dropout方法来避免过拟合,增强泛化能力,
# 不懂的话请参考来理解一下,具体怎么实现不需要特别清楚,  http://blog.csdn.net/zjm750617105/article/details/51313825
dropout = 0.5

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

'''
    conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
           data_format=None, name=None):
           input: [batch, image_size, imgage_size, channels] 其中channel是颜色通道,R,G,B 三通道
           filter: [filter_height, filter_width, in_channels, out_channels], 滤波器就是卷积核kernel
           stride: [1,k,k,1], 先横向步长是k, 然后再枞向步长是k
    biases: 跟truncated_normal的第四个参数一致,一个feature map对应一个bias
    padding : SAME 是宽卷积,就是维数跟原来保持一样, VALID是窄卷积,就是卷积之后实际的维数
            padding=same就相当于在外围补了一圈0，确保你的核中心可以从图的第一个像素点开始移动
            所以一般的都设置为 'SAME'
    ksize:
'''
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    #每次卷机之后进行一次relu激活, 进行一次非线性操作
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper, 下采样操作kzise就是卷积核
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
'''
网络结构:
28*28 -->5*5卷积核,步长是1--> 32张24*24的feature map --> relu(conv1)--> max pooling, ksize 是2*2, 步长是2--> 32张12*12的feature map
--> 5*5卷积核,步长是1--> 64张12*12的feature map --> relu(conv2)--> max pooling, ksize 是2*2, 步长是2 --> 64张7*7的feature map
--> full connection layer, 合并成一维(64*7*7) x Wd(64*7*7 x 1024)--> relu(全联接层的输出是1024维)--> 输出层: 1024 * [1024,10] = 10

'''
# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture, 卷积的输入要求是四维tensor,看上面conv2d()中参数的要求
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout, 在全联接层使用dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop), 每个batch反向调整一次
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        #每隔10个step显示一次结果,这个地方想看的更加清楚,可以把display_step改为1或者是去掉这个if
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.})
