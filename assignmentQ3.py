import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

NUM_FEATURES = 8

learning_rate = 0.5e-06
beta = 1e-3
epochs = 500
batch_size = 32
n_folds=5
#num_neuron in hidden layer
num_neurons = [20,40,60,80,100]

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
np.random.shuffle(cal_housing)
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()

#0.3 times X_data
m = 3* X_data.shape[0] // 10

testX, testY = X_data[:m], Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

mean = np.mean(X_data,axis=0)
stdv = np.std(X_data,axis=0)

trainX = (trainX - mean)/ stdv
testX = (testX - mean)/ stdv

trainNum = trainX.shape[0]
testNum = testX.shape[0]

# experiment with small datasets
#trainX = trainX[:1000]
#trainY = trainY[:1000]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])

#y_ is d
y_ = tf.placeholder(tf.float32, [None, 1])

#errors at all hidden layers
all_hiddenL_err=[]

for num_neuron in num_neurons:
	print("num_neuron: {}".format(num_neuron))

	# Build the graph for the deep net
	W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights1')
	B1 = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32, name='biases1')
	
	W2 = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32), name='weights2')
	B2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases2')
	
	L1 = tf.matmul(x,W1) + B1
	H1 = tf.nn.relu(L1)
	
	y = tf.matmul(H1,W2) + B2
	
	regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
	
	loss = tf.reduce_mean(tf.square(y_ - y)) + (beta*regularizer)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	error = tf.reduce_mean(tf.square(y_ - y))

	nf = trainNum // n_folds
	all_folds_err = []

	for fold in range(n_folds):
		start, end = fold*nf,(fold+1)*nf
		print('Testing fold is: %d, start: %g, stop: %g' % (fold, start, end))
		x_test, y_test = trainX[start:end], trainY[start:end]
		x_train = np.append(trainX[:start], trainX[end:], axis=0)
		y_train = np.append(trainY[:start], trainY[end:], axis=0)
		#fold_test_err=[]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			idx = np.arange(trainNum)

			for i in range(epochs):
				np.random.shuffle(idx)
				x_batch = trainX[idx]
				d_batch = trainY[idx]

				for start, end in zip(range(0, trainNum, batch_size), range(batch_size, trainNum, batch_size)):
					train_op.run(feed_dict={x: x_batch[start:end], y_: d_batch[start:end]})
				print("epoch")
				print(i)

			#returns one error for one fold
			errTest = loss.eval(feed_dict={x: x_test, y_: y_test})
			print("Error for this fold:")
			print(errTest)
			#appends error to list of errors for 5 folds
			all_folds_err.append(errTest)

	# ALL errors for 5 folds
	print("k fold errors for number of neurons in hidden layer {}".format(num_neuron))
	print(all_folds_err)

	all_folds_err_means = np.mean(all_folds_err)

	print(all_folds_err_means)
	all_hiddenL_err.append(all_folds_err_means)

print("ALL error means for all the different number of hidden neurons")
print(all_hiddenL_err)

plt.figure(1)
plt.title("cv errors against number of neurons in hidden layers")
plt.errorbar(np.array(range(len(num_neurons))),np.array(all_hiddenL_err), marker='+', c='r')
plt.xlabel('num_neurons')
plt.ylabel('Cross-Validation Error')
plt.xticks(np.array(range(len(num_neurons))),np.array(num_neurons),rotation="horizontal")
plt.grid()
plt.savefig('Qn3fig1.png')

minErrArg = np.argmin(all_hiddenL_err)
print("int(minErrArg")
print(int(minErrArg))
optimal_num_neuron = num_neurons[int(minErrArg)]
# Build the graph for the deep net
W1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, optimal_num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32),
                 name='weights1')
B1 = tf.Variable(tf.zeros([optimal_num_neuron]), dtype=tf.float32, name='biases1')

W2 = tf.Variable(tf.truncated_normal([optimal_num_neuron, 1], stddev=1.0 / np.sqrt(optimal_num_neuron), dtype=tf.float32),
                 name='weights2')
B2 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases2')

L1 = tf.matmul(x, W1) + B1
H1 = tf.nn.relu(L1)

y = tf.matmul(H1, W2) + B2

regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)

loss = tf.reduce_mean(tf.square(y_ - y)) + (beta * regularizer)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
error = tf.reduce_mean(tf.square(y_ - y))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_err = []
	idx = np.arange(trainNum)

	for i in range(epochs):
		np.random.shuffle(idx)
		x_batch = trainX[idx]
		d_batch = trainY[idx]

		for start,end in zip(range(0,trainNum,batch_size),range(batch_size,trainNum,batch_size)):
			train_op.run(feed_dict={x:x_batch[start:end], y_:d_batch[start:end]})

		errTest = loss.eval(feed_dict={x:testX, y_:testY})
		test_err.append(errTest)

		if i % 5 == 0:
			print('iteration %d, test error: %g' % (i, test_err[i]))

	plt.figure(2)
	plt.title("mean square test error vs epochs")
	plt.plot(range(epochs), test_err, label='Test Error')
	plt.xlabel('Epochs')
	plt.ylabel('Test Error')
	plt.legend(loc='upper right')
	plt.savefig('Qn3fig_b.png')
	plt.show()