#
# Project 2, starter code Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 500
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
	with open(file, 'rb') as fo:
		try:
			samples = pickle.load(fo)
		except UnicodeDecodeError:  # python 3.x
			fo.seek(0)
			samples = pickle.load(fo, encoding='latin1')

	data, labels = samples['data'], samples['labels']

	data = np.array(data, dtype=np.float32)
	labels = np.array(labels, dtype=np.int32)

	labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
	labels_[np.arange(labels.shape[0]), labels - 1] = 1

	return data, labels_


def cnn(images):
	images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

	# Conv layer 1
	conv1_w = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, 50], stddev=np.sqrt(2) / np.sqrt(NUM_CHANNELS * 9 * 9)),
	                      name='weights_1')
	conv1_b = tf.Variable(tf.zeros([50]), name='biases_1')
	conv_1 = tf.nn.relu(tf.nn.conv2d(images, conv1_w, [1, 1, 1, 1], padding='VALID') + conv1_b)

	# First pooling layer
	pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

	# Conv layer 2
	conv2_w = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=np.sqrt(2) / np.sqrt(50 * 5 * 5)), name='weights_2')
	conv2_b = tf.Variable(tf.zeros([60]), name='biases_2')
	conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, conv2_w, [1, 1, 1, 1], padding='VALID') + conv2_b)

	# Second pooling layer
	pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

	dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
	pool_2_flat = tf.reshape(pool_2, [-1, dim])

	# Fully connected layer
	full_w = tf.Variable(tf.truncated_normal([dim, 300], stddev=np.sqrt(2) / np.sqrt(dim)), name='weights_full')
	full_b = tf.Variable(tf.zeros([300]), name='biases_full')

	full_layer = tf.nn.relu(tf.matmul(pool_2_flat, full_w) + full_b)

	# Softmax layer
	softmax_w = tf.Variable(tf.truncated_normal([300, 10], stddev=np.sqrt(2) / np.sqrt(300)), name='weights_softmax')
	softmax_b = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_softmax')

	logits = tf.matmul(full_layer, softmax_w) + softmax_b

	return logits, conv_1, pool_1, conv_2, pool_2


def main():
	trainX, trainY = load_data('data_batch_1')
	print(trainX.shape, trainY.shape)

	testX, testY = load_data('test_batch_trim')
	print(testX.shape, testY.shape)

	trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)

	# Create the model
	x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

	logits, conv_1, pool_1, conv_2, pool_2 = cnn(x)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
	loss = tf.reduce_mean(cross_entropy)

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
	correct_prediction = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	trainNum = len(trainX)
	idx = np.arange(trainNum)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		train_loss = []
		test_acc = []

		for e in range(epochs):
			np.random.shuffle(idx)
			trainX_, trainY_ = trainX[idx], trainY[idx]

			for start, end in zip(range(0, trainNum, batch_size), range(batch_size, trainNum, batch_size)):
				train_step.run(feed_dict={x: trainX_[start:end], y_: trainY_[start:end]})

			train_loss.append(loss.eval(feed_dict={x: trainX_, y_: trainY_}))
			test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
			#print('iter %d: test accuracy %g' % (e, test_acc[e]))
			print('iter %d: test accuracy %g, training loss %g' % (e, test_acc[e], train_loss[e]))

		plt.figure(1)
		plt.plot(range(epochs), train_loss, 'r', label='Training Cost')
		plt.plot(range(epochs), test_acc, 'b', label='Test Accuracy')
		plt.title('Training Cost and Test Accuracy against Epochs')
		plt.xlabel('Epochs')
		plt.ylabel('Cost and Accuracy')
		plt.legend(loc='best')
		plt.savefig('p1q1a.png')

		ind = np.random.randint(0, 2000,2)
		#X = trainX[ind, :]

		for i in range(2):
			X = testX[ind[i],:]

			plt.figure()
			plt.gray()
			X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
			plt.axis('off')
			plt.imshow(X_show)
			plt.savefig('./p1q1b_testPattern{}.png'.format(i))

			conv_1_, pool_1_, conv_2_, pool_2_ = sess.run([conv_1, pool_1, conv_2, pool_2],{x: X.reshape(1, 3 * 32 * 32)})

			plt.figure()
			plt.gray()
			conv_1_ = np.array(conv_1_)
			#number of filters is 50 for first conv layer
			for j in range(50):
				plt.subplot(5, 10, j + 1)
				plt.axis('off')
				plt.imshow(conv_1_[0, :, :, j])
			plt.savefig('./p1q1b_conv_1{}.png'.format(i))

			plt.figure()
			plt.gray()
			pool_1_ = np.array(pool_1_)
			for j in range(50):
				plt.subplot(5, 10, j + 1)
				plt.axis('off')
				plt.imshow(pool_1_[0, :, :, j])
			plt.savefig('./p1q1b_pool_1{}.png'.format(i))

			plt.figure()
			plt.gray()
			conv_2_ = np.array(conv_2_)
			#number of filters is 60 for second conv layer
			for j in range(60):
				plt.subplot(6, 10, j + 1)
				plt.axis('off')
				plt.imshow(conv_2_[0, :, :, j])
			plt.savefig('./p1q1b_conv_2{}.png'.format(i))

			plt.figure()
			plt.gray()
			pool_2_ = np.array(pool_2_)
			for j in range(60):
				plt.subplot(6, 10, j + 1)
				plt.axis('off')
				plt.imshow(pool_2_[0, :, :, j])
			plt.savefig('./p1q1b_pool_2{}.png'.format(i))

if __name__ == '__main__':
	main()
