#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import time


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
batch_size = 16
beta = 1e-6
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
train_Y[train_Y == 7] = 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix

#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)
testX = scale(testX, np.min(testX, axis = 0), np.max(testX,axis=0))
test_Y[test_Y == 7] = 6

testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1 #one hot matrix


# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

n = trainX.shape[0]

all_test_acc = []
# training error using 1- trining accuracy
all_tr_err = []
# training erorr using loss function
all_tr_err2 =[]
all_time = []

hidden_layer_neuron_num = [5,10,15,20,25]
for k in range(len(hidden_layer_neuron_num)):
    print('number of neurons: %d' %(hidden_layer_neuron_num[k]))
    num_neurons = hidden_layer_neuron_num[k]

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
        test_acc = []
        train_err = []
        train_err2 = []
        idx = np.arange(n)
        start_time = time.time()
        for i in range(epochs):

            np.random.shuffle(idx)
            trainX_idx, trainY_idx = trainX[idx], trainY[idx]

            for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):  
                train_op.run(feed_dict={x: trainX_idx[start:end], y_: trainY_idx[start:end]})

            train_err.append(1- accuracy.eval(feed_dict={x: trainX_idx, y_: trainY_idx}))    
            train_err2.append(loss.eval(feed_dict={x: trainX_idx, y_: trainY_idx}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))  
            if i % 100 == 0:
                print('iter %d: accuracy %g: error1 %g: error2 %g' %(i, test_acc[i],train_err[i],train_err2[i]))
        end_time = time.time()
        time_taken = (end_time - start_time)/epochs
        print("time taken per epochs with %d neuron: %g" %(num_neurons,time_taken))
    all_time.append(time_taken)        
    all_test_acc.append(test_acc)
    all_tr_err.append(train_err)
    all_tr_err2.append(train_err2)
    print("\n")

# plot learning curves
plt.figure(1)
plt.plot(range(epochs), all_tr_err[0], label = '5')
plt.plot(range(epochs), all_tr_err[1], label = '10')
plt.plot(range(epochs), all_tr_err[2], label = '15')
plt.plot(range(epochs), all_tr_err[3], label = '20')
plt.plot(range(epochs), all_tr_err[4], label = '25')
plt.legend(loc='upper right', title="num of neurons")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('train error')
plt.title('training error using accuracy fx vs. num of neurons')
plt.savefig('./figures/a_3_1.png')

plt.figure(2)
plt.plot(range(epochs), all_test_acc[0], label = '5')
plt.plot(range(epochs), all_test_acc[1], label = '10')
plt.plot(range(epochs), all_test_acc[2], label = '15')
plt.plot(range(epochs), all_test_acc[3], label = '20')
plt.plot(range(epochs), all_test_acc[4], label = '25')
plt.legend(loc='lower right', title="num of neurons")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('test accuracy')
plt.title('testing accuracy vs. num of neurons')
plt.savefig('./figures/a_3_2.png')

plt.figure(3)
plt.plot(hidden_layer_neuron_num, all_time)
plt.xlabel('Number of Neurons')
plt.ylabel('Time taken per epoch (sec)')
plt.title('epoch time vs. num of neuron')
plt.savefig('./figures/a_3_3.png')

plt.figure(4)
plt.plot(range(epochs), all_tr_err2[0], label = '5')
plt.plot(range(epochs), all_tr_err2[1], label = '10')
plt.plot(range(epochs), all_tr_err2[2], label = '15')
plt.plot(range(epochs), all_tr_err2[3], label = '20')
plt.plot(range(epochs), all_tr_err2[4], label = '25')
plt.legend(loc='upper right', title="num of neurons")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('train error')
plt.title('training error using loss fx vs. num of neuron')
plt.savefig('./figures/a_3_4.png')

plt.show()

