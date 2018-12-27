import tensorflow as tf

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

adder=a+b

with tf.Session() as sess:
	print(sess.run(adder,{a:[1,2,3],b:[5,6,7]}))

