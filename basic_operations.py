#coding: utf-8
'''
Basic Operations example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print "a=2, b=3"
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i" % sess.run(a*b)

#返回一个vairiable的op,先定义好变量空间
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
c = tf.placeholder(tf.int16, shape=[None,2]) #[None, 2] 表示的是任意行 * 2列 的 tensor

# Define some operations
add = tf.add(a, b)
mul = tf.mul(a, b)
add2 = tf.add(a,c)

c_list = numpy.array([[1,2],[1,2]])  #一定要这么写[[1,2]],是两层的表示(1,2)的形状

with tf.Session() as sess:
    # sess.run(param1, feed_dict), param是指的前边定义的一下op(function), 比如add, mul, add2, cost, optima等,
    # 第二个参数是这连个op(function)所需要的参数
    print "Addition with variables: ",  sess.run([add, mul], feed_dict={a: 2, b: 3})
    #print "Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3})
    print "addition with number and metrix", sess.run(add2, feed_dict={a:2, c: c_list})


# ----------------
# More in details:
# Matrix Multiplication from TensorFlow official tutorial

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    result = sess.run(product)
    print result
    # ==> [[ 12.]]

'''
下面是一个实例化变量操作的例子
'''
W = tf.Variable(tf.truncated_normal([2,3],stddev=1))
X = tf.Variable(tf.truncated_normal([3,2],stddev=1))
B = tf.constant(1.)
result = tf.matmul(W,X)+B

with tf.Session() as sess:
    tf.initialize_all_variables().run() #或者写成 sess.run(tf.initialize_all_variables())
    print sess.run(result)
'''
lstm中的输入shape的调整
[batchsize, time_step, n_input] = [2,3,3] , 其3*3是一张图片大小
'''
n_steps = 3
n_input = 3
#a的shape = [batch_size, n_steps, n_input]
a = [[[1,1,1],[2,2,2],[3,3,3]],[[1,1,1],[2,2,2],[3,3,3]]]
#b.shape = [n_steps, batch-size, n_input]
b= tf.transpose(a,[1,0,2])
sess = tf.Session()
print  sess.run(b)
'''
array([[[1, 1, 1],
        [1, 1, 1]],

       [[2, 2, 2],
        [2, 2, 2]],

       [[3, 3, 3],
        [3, 3, 3]]], dtype=int32)
'''
c =  tf.reshape(b, [-1, n_input])   #[n_steps*batch-size, n_input]
sess.run(c)
'''
array([[1, 1, 1],
       [1, 1, 1],
       [2, 2, 2],
       [2, 2, 2],
       [3, 3, 3],
       [3, 3, 3]], dtype=int32)
'''
d = tf.split(0, n_steps, c) #[n_steps, batch_size, n_input]
sess.run(d)
'''
[array([[1, 1, 1],
       [1, 1, 1]], dtype=int32), array([[2, 2, 2],
       [2, 2, 2]], dtype=int32), array([[3, 3, 3],
       [3, 3, 3]], dtype=int32)]
'''
