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

# 3-layer neural network
def ffn1(x, hidden_units):
  
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

# 4-layer neural network
def ffn2(x, hidden1_units, hidden2_units):
  
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights1 = tf.Variable(
      tf.truncated_normal([NUM_FEATURES, hidden1_units],
                            stddev=1.0 / np.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases)
    

  # Hidden 2
  with tf.name_scope('hidden2'):
    weights2 = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / np.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases)

  # Linear
  with tf.name_scope('softmax_linear'):
    weights3 = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / np.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights3) + biases
    
  return weights1,weights2,weights3,logits

NUM_FEATURES = 36
NUM_CLASSES = 6

learning_rate = 0.01
epochs = 1000
batch_size = 32
num_neurons1 = 10
num_neurons2 = 10
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
all_train_err1 = []
all_train_err2 = []
all_test_acc = []
all_train_acc = []

for k in range(2):
    print(k)
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    if(k==0):
        w1, w2, y = ffn1(x, num_neurons1)
    else:	
        w1, w2, w3, y = ffn2(x, num_neurons1, num_neurons2)	
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
  
    if(k==0):
        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    else:
        regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    loss = tf.reduce_mean(cross_entropy + beta*regularization)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        train_error1 = []
        train_error2 = []
        test_acc = []
        idx = np.arange(n)

        for i in range(epochs):

            np.random.shuffle(idx)
            trainX_idx, trainY_idx = trainX[idx], trainY[idx]

            for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):  
                train_op.run(feed_dict={x: trainX_idx[start:end], y_: trainY_idx[start:end]})
    
            train_acc_temp = accuracy.eval(feed_dict={x: trainX_idx, y_: trainY_idx})
            train_acc.append(train_acc_temp)
            train_error1.append(1 - train_acc_temp)
            train_error2.append(loss.eval(feed_dict={x: trainX_idx, y_: trainY_idx}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))  
            if i % 100 == 0:
                print('iter %d: tr_acc %g: tr_er1 %g: tr_er2 %g: test_acc %g'%(i, train_acc[i],train_error1[i],train_error2[i],test_acc[i]))


        all_test_acc.append(test_acc)
        all_train_acc.append(train_acc)
        all_train_err1.append(train_error1)
        all_train_err2.append(train_error2)





# plot learning curves
plt.figure(1)
plt.plot(range(epochs), all_train_acc[0], label="3-layer")
plt.plot(range(epochs), all_train_acc[1], label="4-layer")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.legend(loc='lower right', title="layer NN")
plt.title('3-layer vs 4-layer')
plt.savefig('./figures/a_5_1.png')

plt.figure(2)
plt.plot(range(epochs), all_train_err1[0], label="3-layer")
plt.plot(range(epochs), all_train_err1[1], label="4-layer")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train error using accuracy fx')
plt.title('3-layer vs 4-layer')
plt.legend(loc='upper right', title="layer NN")
plt.savefig('./figures/a_5_2.png')

plt.figure(3)
plt.plot(range(epochs), all_train_err2[0], label="3-layer")
plt.plot(range(epochs), all_train_err2[1], label="4-layer")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train error using loss function')
plt.title('3-layer vs 4-layer')
plt.legend(loc='upper right', title="layer NN")
plt.savefig('./figures/a_5_3.png')

plt.figure(4)
plt.plot(range(epochs), all_test_acc[0], label = "3-layer")
plt.plot(range(epochs), all_test_acc[1], label = "4-layer")
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test accuracy')
plt.title('3-layer vs 4-layer')
plt.legend(loc='lower right', title="layer NN")
plt.savefig('./figures/a_5_4.png')


plt.show()

