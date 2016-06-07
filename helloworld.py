'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf

#Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
with tf.Session() as sess:

    # Run the op
    '''
    tf中所有的操作都可以看作是op或者是由op组成的,所有的op都需要通过sess来运行
    '''
    print sess.run(hello)
