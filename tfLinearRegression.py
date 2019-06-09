import tensorflow as tf 
import matplotlib.pyplot as plt

tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)
slope = tf.Variable(0.1)
intercept = tf.Variable(0.5)

model = slope * input_data + intercept
sq_error = tf.square(model - output_data)
loss = tf.reduce_mean(sq_error)

x_values = [0,1,2,3,4,5]
y_values = [0.5,3,5.2,7.2,8,10.5]

optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, feed_dict={input_data:x_values, output_data:y_values})

        plt.plot(x_values, y_values, 'ro', 'Training Data')
        plt.plot(x_values, sess.run(model, feed_dict={input_data: x_values}), 'b')

    plt.plot(x_values, sess.run(model, feed_dict={input_data: x_values}), 'r')


plt.show()

