import tensorflow as tf

W=tf.Variable([0.2],tf.float32)
b=tf.Variable([-0.2],tf.float32)

x=tf.placeholder(tf.float32)

linear_model= W*x+b

y=tf.placeholder(tf.float32)

#loss
square_delta=tf.square(linear_model-y)
loss=tf.reduce_sum(square_delta)

#Optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
	print(sess.run([W,b]))



