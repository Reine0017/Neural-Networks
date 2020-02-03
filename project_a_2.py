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

def main():
    tf.set_random_seed(seed)

    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    w1, w2, y = ffn(x, num_neurons)	

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)

    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)

    loss = tf.reduce_mean(cross_entropy + beta*regularization)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), dtype=tf.int32))

  
    batch_sizes = [4, 8, 16, 32, 64]

    all_test_acc = []
    # training error using 1- trining accuracy
    all_tr_err = []
    # training erorr using loss function
    all_tr_err2 =[]
    all_time = []

    for j in range(len(batch_sizes)):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = batch_sizes[j]
            idx = np.arange(n)
            test_acc = []
            train_err = []
            train_err2 = []
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
                    print('batch %d, epoch %d, error1 %g, error2 %g, accuracy %g'%(batch_size, i, train_err[i], train_err2[i], test_acc[i]))
            
            end_time = time.time()
            time_taken = (end_time - start_time)/epochs
            print("time taken per epochs batch %d: %g" %(batch_size,time_taken))
            all_time.append(time_taken)        
            all_test_acc.append(test_acc)
            all_tr_err.append(train_err)
            all_tr_err2.append(train_err2)
   

    # print(all_test_acc)
    # print(all_tr_err)

    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), all_tr_err[0], label = '4')
    plt.plot(range(epochs), all_tr_err[1], label = '8')
    plt.plot(range(epochs), all_tr_err[2], label = '16')
    plt.plot(range(epochs), all_tr_err[3], label = '32')
    plt.plot(range(epochs), all_tr_err[4], label = '64')
    plt.legend(loc='upper right', title="batch sizes")
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('train error')
    plt.title('training error using accuracy fx vs. batch size')
    plt.savefig('./figures/a_2_1.png')

    plt.figure(2)
    plt.plot(range(epochs), all_test_acc[0], label = '4')
    plt.plot(range(epochs), all_test_acc[1], label = '8')
    plt.plot(range(epochs), all_test_acc[2], label = '16')
    plt.plot(range(epochs), all_test_acc[3], label = '32')
    plt.plot(range(epochs), all_test_acc[4], label = '64')
    plt.legend(loc='lower right', title="batch sizes")
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('test accuracy')
    plt.title('testing accuracy vs. batch size')
    plt.savefig('./figures/a_2_2.png')

    plt.figure(3)
    plt.plot(batch_sizes, all_time)
    plt.xlabel('Batch Size')
    plt.ylabel('Time taken per epoch (sec)')
    plt.title('time to epoch vs. batch size')
    plt.savefig('./figures/a_2_3.png')
    
    plt.figure(4)
    plt.plot(range(epochs), all_tr_err2[0], label = '4')
    plt.plot(range(epochs), all_tr_err2[1], label = '8')
    plt.plot(range(epochs), all_tr_err2[2], label = '16')
    plt.plot(range(epochs), all_tr_err2[3], label = '32')
    plt.plot(range(epochs), all_tr_err2[4], label = '64')
    plt.legend(loc='upper right', title="batch sizes")
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('train error')
    plt.title('training error using loss fx vs. batch size')
    plt.savefig('./figures/a_2_4.png')


    # plt.plot(range(epochs), train_acc)
    # plt.xlabel(str(epochs) + ' iterations')
    # plt.ylabel('Train accuracy')
    plt.show()

if __name__ == '__main__':
  main()


