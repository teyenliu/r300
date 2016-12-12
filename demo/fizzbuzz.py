'''
Created on Dec 3, 2016

@author: liudanny
'''

# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf
import os
import json

NUM_DIGITS = 10
# for example:/home/liudanny/workspace/r300/demo/
LIC_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_h2, w_o, b):
    h = tf.nn.relu(tf.matmul(X, w_h))
    h2 = tf.nn.relu(tf.matmul(h, w_h2))
    return tf.matmul(h2, w_o)

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_h2 = init_weights([NUM_HIDDEN, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])
b = tf.Variable(tf.zeros([1, 1]))

# Predict y given x using the model.
py_x = model(X, w_h, w_h2, w_o, b)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 128


class PredictService(object):
    def __init__(self, checkpoint_path, checkpoint_file):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file = checkpoint_file
        self.sess = None
        #self.inputs = None
        #self.outputs = None

        self.init_session_handler()

    def init_session_handler(self):
        self.sess = tf.Session()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Use the model {}".format(ckpt.model_checkpoint_path))
            saver = tf.train.import_meta_graph(self.checkpoint_file)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            #self.inputs = json.loads(tf.get_collection('inputs')[0])
            #self.outputs = json.loads(tf.get_collection('outputs')[0])
        else:
            print("No model found, exit now")
            exit()

    def predict(self, num):
        num_ndarray = np.array([num])
        teX = np.transpose(binary_encode(num_ndarray, NUM_DIGITS))
        teY = self.sess.run(predict_op, feed_dict={X: teX})
        result =  fizz_buzz(num, teY)
        print("Request number: %i, result: %s" % (num, result))
        return result


"""
def predict(num):
    # Launch the graph in a session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print ckpt_dir
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        
        num_ndarray = np.array([num])
        teX = np.transpose(binary_encode(num_ndarray, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        return fizz_buzz(num, teY)
    
def train():
    # Launch the graph in a session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

        for epoch in range(10000):
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            # Train in batches of 128 inputs.
            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            # And print the current accuracy on the training data.
            if epoch % 100 == 0 :
                saver.save(sess, ckpt_dir + "/model.ckpt", global_step=epoch)
            print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))

        # And now for some fizz buzz
        numbers = np.arange(1, 101)
        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        print(output)
"""