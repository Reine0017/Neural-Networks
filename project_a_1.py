#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def ffn(x, hidden_units):
  
  # Hidden
  with tf.name_scope('hidden'):
    weights1 = tf.Variable(
      tf.truncated_normal([NUM_FEATURES, hidden_units],
                            stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),
                         name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights1) + biases)
    
  # Linear
  with tf.name_scope('softmax_linear'):
    weights2 = tf.Variable(
        tf.truncated_normal([hidden_units, NUM_CLASSES],
                            stddev=1.0 / np.sqrt(float(hidden_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden, weights2) + biases
    
  return weights1,weights2,logits

NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons = 10
beta = 10**(-6)
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix


# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

n = trainX.shape[0]


# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# Build the graph for the deep net
w1, w2, y = ffn(x, num_neurons)	
# weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
# biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
# logits  = tf.matmul(x, weights) + biases

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
# loss = tf.reduce_mean(cross_entropy)

regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

loss = tf.reduce_mean(cross_entropy + beta*regularization)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []

    idx = np.arange(n)

    for i in range(epochs):

        np.random.shuffle(idx)
        trainX_idx, trainY_idx = trainX[idx], trainY[idx]

        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):  
            train_op.run(feed_dict={x: trainX_idx[start:end], y_: trainY_idx[start:end]})
            
        train_acc.append(accuracy.eval(feed_dict={x: trainX_idx, y_: trainY_idx}))  
        if i % 5 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))


# plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

