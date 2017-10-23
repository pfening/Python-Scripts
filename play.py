# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

matrix1 = tf.constant([[2.,4.]])
matrix2 = tf.constant([[2.],[3.]])

product = tf.nn.relu(tf.matmul(matrix1, matrix2))

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))
print(sess.run(product))
# Close the session
sess.close()