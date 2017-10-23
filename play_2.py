import tensorflow as tf
import numpy as np

DTYPE = tf.float32

## Generate Data
age = np.array([67, 38, 32, 69, 40])
awesomeness = 10 * age

## Generate model
# define the parameter of the model
slope = tf.Variable(initial_value=tf.random_normal(shape=(1,), dtype=DTYPE))
# define the data inputs to the model as variable size tensors
x = tf.placeholder(DTYPE, shape=(None,))
y_data = tf.placeholder(DTYPE, shape=(None,))
# specify the model
y_pred = slope * x
# use mean squared error as loss function
loss = tf.reduce_mean(tf.square(y_data - y_pred))
target = tf.train.AdamOptimizer().minimize(loss)

## Train Model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(25000):
        _, training_loss = sess.run([target, loss],
                                    feed_dict={x: age, y_data: awesomeness})
    print("Training loss: ", training_loss)
    print("Found slope=", sess.run(slope))