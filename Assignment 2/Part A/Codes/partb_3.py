import numpy as np
import pandas
import tensorflow as tf
import pylab as plt
import csv

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
EMBEDDING_SIZE = 20
MAX_LABEL = 15
batch_size = 128

no_epochs = 1000
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_rnn_model(x):

  byte_vectors = tf.one_hot(x, 256)
  byte_list = tf.unstack(byte_vectors, axis=1)


  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, byte_list


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

  
def main():
  
  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, byte_list = char_rnn_model(x)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)


  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.one_hot(y_, MAX_LABEL),1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  N = len(x_train)
  idx = np.arange(N)
  

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  loss = []
  test_acc = []
  for i in range(no_epochs):
    np.random.shuffle(idx)
    trainX, trainY = x_train[idx], y_train[idx]

    for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
      sess.run(train_op, {x: trainX[start:end], y_: trainY[start:end]})
 
    loss_  = sess.run(entropy, {x: trainX, y_: trainY})
    loss.append(loss_)
    accuracy_ = sess.run(accuracy, {x: x_test, y_: y_test})
    test_acc.append(accuracy_)
    print('iter: %d, entropy: %g, accuracy: %g'%(i, loss[i],test_acc[i]))
  
  # plot learning curves
  plt.figure(1)
  plt.plot(range(no_epochs), test_acc)
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('Test Accuracy')
  plt.title('Testing Accuracy VS. Epochs')
#   plt.savefig('./figures/p2_b1_1.png')
  
  # plot learning curves
  plt.figure(2)
  plt.plot(range(no_epochs), loss, label = 'Entropy Cost')
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('Entropy Cost')
  plt.title('Cost Entropy VS. epochs')
#   plt.savefig('./figures/p2_b1_2.png')

  plt.show()



if __name__ == '__main__':
  main()
