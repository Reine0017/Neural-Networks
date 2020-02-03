import tensorflow as tf
import numpy as np
import pylab as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

NUM_FEATURES = 8
keep_prob = 0.9
learning_rate = 1e-9
beta = 1e-3
epochs = 500
batch_size = 32
n_folds = 5
# num_neuron in hidden layer
num_neuron1 = 60
num_neuron2 = 20

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()

print(X_data)
print(Y_data)

# 0.3 times X_data
m = 3 * X_data.shape[0] // 10

testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

mean = np.mean(X_data, axis=0)
stdv = np.std(X_data, axis=0)

trainX = (trainX - mean) / stdv
testX = (testX - mean) / stdv

trainNum = trainX.shape[0]
testNum = testX.shape[0]

# experiment with small datasets
# trainX = trainX[:1000]
# trainY = trainY[:1000]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])

# y_ is d
y_ = tf.placeholder(tf.float32, [None, 1])

# Build the graph for the deep net
W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32),
                 name='weights1')
B1 = tf.Variable(tf.zeros([num_neuron1]), dtype=tf.float32, name='biases1')

W2 = tf.Variable(tf.truncated_normal([num_neuron1, num_neuron2], stddev=1.0 / np.sqrt(num_neuron1), dtype=tf.float32),
                 name='weights2')
B2 = tf.Variable(tf.zeros([num_neuron2]), dtype=tf.float32, name='biases2')

W3 = tf.Variable(tf.truncated_normal([num_neuron2, 1], stddev=1.0 / np.sqrt(num_neuron2), dtype=tf.float32),
                 name='weights3')
B3 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases2')

# L1 = tf.matmul(x,W1) + B1
# H1 = tf.nn.relu(L1)
H1 = tf.nn.relu(tf.matmul(x, W1) + B1)
H2 = tf.nn.relu(tf.matmul(H1, W2) + B2)
y_None = tf.matmul(H2, W3) + B3

H1 = tf.nn.relu(tf.matmul(x, W1) + B1)
H1_dropout = tf.nn.dropout(H1,keep_prob)
H2 = tf.nn.relu(tf.matmul(H1_dropout, W2) + B2)
H2_dropout = tf.nn.dropout(H2,keep_prob)
y_drop = tf.matmul(H2_dropout, W3) + B3

myList = [y_None,y_drop]
errList=[]
rmseList=[]

errListD=[]
rmseListD=[]

count=0

for y in myList:
	#print(y)

	regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)

	loss = tf.reduce_mean(tf.square(y_ - y)) + (beta * regularizer)

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))

	train_err = []
	test_err = []

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		idx = np.arange(trainNum)

		for i in range(epochs):
			np.random.shuffle(idx)
			x_batch = trainX[idx]
			d_batch = trainY[idx]

			for start, end in zip(range(0, trainNum, batch_size), range(batch_size, trainNum, batch_size)):
				train_op.run(feed_dict={x: x_batch[start:end], y_: d_batch[start:end]})

			errTrain = loss.eval(feed_dict={x: trainX, y_: trainY})
			train_err.append(errTrain)

			errTest = loss.eval(feed_dict={x: testX, y_: testY})
			test_err.append(errTest)

			if i % 50 == 0:
				print('iteration %d, error: train %g, test %g' % (i, train_err[i], test_err[i]))

		RMSE_test = np.sqrt(test_err)

	if count==0:
		errList.append(test_err)
		rmseList.append(RMSE_test)
	else:
		errListD.append(test_err)
		rmseListD.append(RMSE_test)

	count = count+1

a = np.reshape(errList,(epochs,))
b = np.reshape(rmseList,(epochs,))
c = np.reshape(errListD,(epochs,))
d = np.reshape(rmseListD,(epochs,))

print(a.shape)
plt.figure(1)
plt.title("mean square error vs epochs")
plt.plot(range(epochs), a, label='Test Error', c='r')
plt.plot(range(epochs), c, label='Test Error with dropouts', c='b')
plt.xlabel('epochs')
plt.ylabel('test Error')
plt.legend(loc='upper right')
plt.savefig('Qn1fig4a_1.png')

plt.figure(2)
plt.title("root mean square error vs epochs")
plt.plot(range(epochs), b, label='Testing Error', c='r')
plt.plot(range(epochs), d, label='Testing Error with dropouts', c='b')
plt.xlabel('epochs')
plt.ylabel('Testing Error')
plt.legend(loc='upper right')
plt.savefig('Qn4fig4a_1_r.png')
plt.show()