import numpy as np
import pandas
import tensorflow as tf
import pylab as plt
import csv


MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
batch_size = 128
is_dropout = False
keep_prob = 0.5
# model can be changed into 'rnn', 'gru', and 'lstm'
model = 'rnn' 
# layer can be changed into 1 and 2
layer = 1
threshold = 2.0

no_epochs = 5000
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_rnn_model(x, is_dropout, keep_prob, model):

  byte_vectors = tf.one_hot(x, 256)
  byte_list = tf.unstack(byte_vectors, axis=1)

  with tf.variable_scope('RNN_1'):

    # choose cell type
    if model == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fn = tf.nn.rnn_cell.LSTMCell

    # multi-layer cell
    if(layer > 1):
        cell1 = cell_fn(HIDDEN_SIZE,reuse = tf.get_variable_scope().reuse)
        cell2 = cell_fn(HIDDEN_SIZE,reuse = tf.get_variable_scope().reuse)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])
    else:
        cell = cell_fn(HIDDEN_SIZE)

    if is_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob= keep_prob,output_keep_prob= keep_prob)

    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    if isinstance(encoding, tuple):
        encoding = encoding[-1]

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits

def word_rnn_model(x, is_dropout, keep_prob, model):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  with tf.variable_scope('RNN_2'):

    # choose cell type
    if model == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fn = tf.nn.rnn_cell.LSTMCell

    # multi-layer cell
    if(layer > 1):
        cell1 = cell_fn(HIDDEN_SIZE,reuse = tf.get_variable_scope().reuse)
        cell2 = cell_fn(HIDDEN_SIZE,reuse = tf.get_variable_scope().reuse)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2])
    else:
        cell = cell_fn(HIDDEN_SIZE)

    if is_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob= keep_prob,output_keep_prob= keep_prob)

    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    if isinstance(encoding, tuple):
        encoding = encoding[-1]

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

    if(case == 'rnn-char'):
        x_train, y_train, x_test, y_test = read_data_chars()
    elif(case == 'rnn-word'):
        x_train, y_train, x_test, y_test, n_words = data_read_words()

    return  x_train, y_train, x_test, y_test, n_words

def rnn_call(case, x, is_dropout, keep_prob, model):
    logits = None

    if(case == 'rnn-char'):
        logits = char_rnn_model(x, is_dropout, keep_prob, model)
    elif(case == 'rnn-word'):
        logits = word_rnn_model(x, is_dropout, keep_prob, model)
    
    return logits

def main():
    global n_words

    list_case = ['rnn-char','rnn-word']
    all_test_acc = []
    
    for c,k in enumerate(list_case):
        print(k)
        x_train, y_train, x_test, y_test, n_words = read_data(k)

        # Create the model
        x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        y_ = tf.placeholder(tf.int64)
        keep_prob = tf.placeholder(tf.float32)

        logits = rnn_call(k,x, is_dropout, keep_prob, model)

        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
        
        # Minimizer
        minimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = minimizer.compute_gradients(entropy)

        # Gradient clipping
        grad_clipping = tf.constant(threshold, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is None:
                clipped_grad = None
            else:
                clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))

        # Gradient updates
        train_op = minimizer.apply_gradients(clipped_grads_and_vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL),1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        N = len(x_train)
        idx = np.arange(N)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_acc = []

        
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                sess.run(train_op, {x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.7})
        
            test_acc_ = sess.run(accuracy, {x: x_test, y_: y_test, keep_prob: 1.0})
            test_acc.append(test_acc_)
        
            print('iter: %d, testacc: %g'%(i, test_acc[i]))
      

        all_test_acc.append(test_acc)
        sess.close()
        

    plt.figure(1)
    plt.plot(range(no_epochs), all_test_acc[0], label = 'rnn-char')
    plt.plot(range(no_epochs), all_test_acc[1], label = 'rnn-word')
    plt.legend(loc='lower right')
    plt.xlabel(str(no_epochs) + ' iterations')
    plt.ylabel('test accuracy')
    plt.title('test accuracy with grad clipping vs. epochs')

    
    plt.show()

if __name__ == '__main__':
  main()
