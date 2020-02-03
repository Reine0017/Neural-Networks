import numpy as np
import pandas
import tensorflow as tf
import pylab as plt
import csv
import time

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
FILTER_SHAPE3 = [20, 20]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
EMBEDDING_SIZE = 20
batch_size = 128
is_dropout = True

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_cnn_model(x, is_dropout, keep_prob):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  
  input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])

  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE3,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if is_dropout:
        pool1 = tf.nn.dropout(pool1,keep_prob)
  with tf.variable_scope('CNN_Layer2'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')  
    if is_dropout:
        pool2 = tf.nn.dropout(pool2,keep_prob)
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

   
  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return logits


def char_cnn_model(x, is_dropout, keep_prob):
  
  input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

  with tf.variable_scope('CNN_Layer3'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if is_dropout:
        pool1 = tf.nn.dropout(pool1,keep_prob)

  with tf.variable_scope('CNN_Layer4'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')  
    if is_dropout:
        pool2 = tf.nn.dropout(pool2,keep_prob)

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

   
  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return logits

def char_rnn_model(x, is_dropout, keep_prob):

  byte_vectors = tf.one_hot(x, 256)
  byte_list = tf.unstack(byte_vectors, axis=1)

  with tf.variable_scope('RNN_1'):
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    if is_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob= keep_prob,output_keep_prob= keep_prob)
    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits

def word_rnn_model(x, is_dropout, keep_prob):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  with tf.variable_scope('RNN_2'):
    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    if is_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob= keep_prob,output_keep_prob= keep_prob)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits


def data_read_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
 
  return x_train, y_train, x_test, y_test


def read_data(case):
    x_train, y_train, x_test, y_test = [], [], [], []
    n_words = 0

    if(case == 'cnn-char'):
        x_train, y_train, x_test, y_test = read_data_chars()    
    elif(case == 'cnn-word'):
        x_train, y_train, x_test, y_test, n_words = data_read_words()
    elif(case == 'rnn-char'):
        x_train, y_train, x_test, y_test = read_data_chars()
    elif(case == 'rnn-word'):
        x_train, y_train, x_test, y_test, n_words = data_read_words()

    return  x_train, y_train, x_test, y_test, n_words

def cnn_rnn_call(case, x, is_dropout, keep_prob):
    logits = None

    if(case == 'cnn-char'):
        logits = char_cnn_model(x, is_dropout, keep_prob)
    elif(case == 'cnn-word'):
        logits = word_cnn_model(x, is_dropout, keep_prob)
    elif(case == 'rnn-char'):
        logits = char_rnn_model(x, is_dropout, keep_prob)
    elif(case == 'rnn-word'):
        logits = word_rnn_model(x, is_dropout, keep_prob)
    
    return logits

def main():
    global n_words

    list_case = ['cnn-char','cnn-word','rnn-char','rnn-word']
    all_test_acc = []
    all_time = []
    for c,k in enumerate(list_case):
        print(k)
        x_train, y_train, x_test, y_test, n_words = read_data(k)

        # Create the model
        x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        y_ = tf.placeholder(tf.int64)
        keep_prob = tf.placeholder(tf.float32)

        logits = cnn_rnn_call(k,x, is_dropout, keep_prob)

        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL),1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        N = len(x_train)
        idx = np.arange(N)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_acc = []

        start_time = time.time()
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                sess.run(train_op, {x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.7})
        
            test_acc_ = sess.run(accuracy, {x: x_test, y_: y_test, keep_prob: 1.0})
            test_acc.append(test_acc_)
        
            print('iter: %d, testacc: %g'%(i, test_acc[i]))
        end_time = time.time()
        time_taken = (end_time - start_time)/no_epochs
        print("time taken per epochs batch %d: %g" %(batch_size,time_taken))

        all_test_acc.append(test_acc)
        all_time.append(time_taken)

    plt.figure(1)
    plt.plot(range(no_epochs), all_test_acc[0], label = 'cnn-char')
    plt.plot(range(no_epochs), all_test_acc[1], label = 'cnn-word')
    plt.plot(range(no_epochs), all_test_acc[2], label = 'rnn-char')
    plt.plot(range(no_epochs), all_test_acc[3], label = 'rnn-word')
    plt.legend(loc='lower right')
    plt.xlabel(str(no_epochs) + ' iterations')
    plt.ylabel('test accuracy')
    plt.title('test accuracy vs. epochs')

    plt.figure(2)
    plt.plot(list_case, all_time)
    plt.legend(loc='lower right')
    plt.ylabel('sec')
    plt.title('time taken per epoch')
    
    plt.show()

if __name__ == '__main__':
  main()
