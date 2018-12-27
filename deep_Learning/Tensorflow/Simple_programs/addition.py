import tensorflow as tf

a=tf.constant(5.0, tf.float32)
b=tf.constant(4.0, tf.float32)

sum=a+b

with tf.Session() as sess:
	print(sess.run(sum))

