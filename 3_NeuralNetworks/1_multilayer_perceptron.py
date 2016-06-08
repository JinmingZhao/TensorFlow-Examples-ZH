'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])  #表示任意个n_input向量
y = tf.placeholder("float", [None, n_classes]) #表示任意个n_classes向量


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#构建训练模型, 返回一个输出层的tensor
pred = multilayer_perceptron(x, weights, biases)

# 定义目标函数为交叉熵函数并求平均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))


'''
下面定义所使用的优化方法(权值调整方法),并分别利用gd(梯度下降) 和 adam 优化算法进行了测试,
测试结果证明,adam整体效果要比gd要好,具体看下面对比结果.

tf支持了所有主流的优化算法:  https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html
'''
'''
gd:
accurancy: 0.9123 cost:12.5 epoch=15
accurancy: 0.9188 cost: 7.5 epoch=25
adam:
accurancy: 0.9454 cost:0.94 epoch=15
accurancy: 0.9539 cost:0.32 epoch=25
'''
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables, 但是必须用sess.run(init)之后才能使用
init = tf.initialize_all_variables()

# 所有上面的都是在构建各种变量,构建模型,都没有进行具体的运算, Launch the graph
with tf.Session() as sess:

    # 或者可以不要上面那个init, 直接tf.initialize_all_variables().run()
    sess.run(init)

    # Training cycle, 所有的样本集训练 training_epochs 次
    for epoch in range(training_epochs):
        avg_cost = 0.
        #求得训练集的batch的个数
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            '''
            每次取出一个batch进行训练, 一个batch包括[batch_x(输入sample), batch_y(labels)],
            sess.run(features, feed_dict)
                features: 前面定义的op(计算的公式模型), 支持多输入多输出
                feed_dict:  求得features的所需要的参数,
                    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
                    看这两个函数需要的参数有 w  b x y ,  而w b, 前边已经显示的定义了,
                    而x, y 由于不确定batch size大小,只是占了个坑而已(placehold)
            '''
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step, 训练集训练一个显示一次
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    '''
    arg_max(input, dimension, name=None)
        input: tensor 或者是 array
        dimension: 1表示在一行中操作, 0 表示在列中操作
    example:
        >>> a = [[1,2],[3,4],[5,6]]
        >>> c = tf.argmax(a, 1) #在每一行中找出最大值的索引
        >>> sess = tf.Session()
        >>> sess.run(c)
        >>> array([1, 1, 1])
        >>> d = tf.argmax(a, 0) #在每一列中找出最大值的索引
        >>> sess = tf.Session()
        >>> sess.run(d)
        >>> array([2,2])
    训练结束之后,需要用测试集对训练好的模型进行测试,由于标签向量是由0,1组成(注意在获取数据集的时候有一个参数 one-hot = ture,
    就是说转化为0,1的数组,存在value的位置是1, 否则是0)，因此最大值1所在的索引位置就是类别标签,
    比如tf.argmax(pred,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y,1)代表正确的标签，
    我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
    '''
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 通过训练好的accuracy模型来训练test数据集,test的batchsize = 1, 这个模型包括correct_prediction这个OP以及所有的子OP
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
