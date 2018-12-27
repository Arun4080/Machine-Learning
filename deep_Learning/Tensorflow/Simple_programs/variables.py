import tensorflow as tf

a=tf.Variable(.3,tf.float32)
b=tf.Variable(.2,tf.float32)

sum=a+b
init=tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(sum))

