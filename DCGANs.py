import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def generator(inputs, reuse=False, training=True):
	with tf.device('/device:GPU:0'):
		s_size = 4
		initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
		with tf.variable_scope('g', reuse=reuse):
			z = tf.convert_to_tensor(inputs)
			# z vector convert into tensor
			outputs = tf.layers.dense(z, s_size * s_size * 1024)
			outputs = tf.reshape(outputs, [-1, s_size, s_size, 1024])
			outputs = tf.layers.batch_normalization(outputs, training=training)
			outputs = tf.nn.relu(outputs) # shape (batch_size, 4, 4, 1024)

			# Transposed conv 1
			tconv1 = tf.layers.conv2d_transpose(inputs=outputs, filters=512, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			tconv1 = tf.layers.batch_normalization(tconv1, training=training)
			tconv1 = tf.nn.relu(tconv1) # shape = (batch_size, 8, 8, 512)

			# Transposed conv 2
			tconv2 = tf.layers.conv2d_transpose(inputs=tconv1, filters=256, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			tconv2 = tf.layers.batch_normalization(tconv2, training=training)
			tconv2 = tf.nn.relu(tconv2) # output shape = (batch_size, 16, 16, 256)

			# Transposed conv 3
			tconv3 = tf.layers.conv2d_transpose(inputs=tconv2, filters=128, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			tconv3 = tf.layers.batch_normalization(tconv3, training=training)
			tconv3 = tf.nn.relu(tconv3) # output shape = (batch_size, 32, 32, 128)

			# Transposed conv 4 Filters == RGB
			tconv4 = tf.layers.conv2d_transpose(inputs=tconv3, filters=3, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			tconv4 = tf.layers.batch_normalization(tconv4, training=training)

			# tanh output
			g = tf.nn.tanh(tconv4, name='generator') # output shape = (batch_size, 64, 64, 3)
			return g


def make_fakes(batch_size=128, row=1, col=8):
	images = generator(z, reuse=True, training=False)
	images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
	images = [image for image in tf.split(images, batch_size, axis=0)]
	rows = []
	for i in range(row):
		rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
	image = tf.concat(rows, 1)
	return image;


def discriminator(inputs, reuse=False, training=True):
	with tf.device('/device:GPU:0'):
		initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
		with tf.variable_scope('d', reuse=reuse):
			d_inputs = tf.convert_to_tensor(inputs)

			# conv 1
			conv1 = tf.layers.conv2d(d_inputs, 64, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv1 = tf.layers.batch_normalization(conv1, training=training)
			conv1 = tf.nn.leaky_relu(conv1) # shape (batch_size, 32, 32, 64)

			# conv 2
			conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv2 = tf.layers.batch_normalization(conv2, training=training)
			conv2 = tf.nn.leaky_relu(conv2) # shape (batch_size, 16, 16, 128)

			# conv 3
			conv3 = tf.layers.conv2d(conv2, 256, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv3 = tf.layers.batch_normalization(conv3, training=training)
			conv3 = tf.nn.leaky_relu(conv3) # shape (batch_size, 4, 4, 256)

			# conv 4
			conv4 = tf.layers.conv2d(conv3, 512, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
			conv4 = tf.layers.batch_normalization(conv4, training=training)
			conv4 = tf.nn.leaky_relu(conv4) # shape (batch_size, 2, 2, 512)

			batch_size = conv4.get_shape()[0].value

			reshape = tf.reshape(conv4, [batch_size, -1])
			d = tf.layers.dense(reshape, 1, name='d')
			d = tf.nn.sigmoid(reshape)

			return d

def trains(batch_size=128, z_dim=100):
	with tf.device('/device:GPU:0'):
		z = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0)
		fake_data = generator(z, reuse=False, training=True)
		input_data = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32) # for mnist
		
		# convert raw data to valid image data (for mnist)
		images = tf.reshape(input_data, (batch_size, 28, 28, 1))
		images = tf.image.resize_images(images, (64,64))
		images = tf.image.grayscale_to_rgb(images)

		d_r = discriminator(images, reuse=False, training=True)
		d_f = discriminator(fake_data, reuse=True, training=True)

		# losses
		d_loss = -tf.reduce_mean( tf.log(d_r) + tf.log(1-d_f))  
		g_loss = tf.reduce_mean(tf.log(1-d_f))

		# optimizer
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # make work batchnorm
		with tf.control_dependencies(update_ops):
			d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
			d_train = d_opt.minimize(d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d'))
			g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
			g_train = g_opt.minimize(g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g'))
			return d_train, g_train, d_loss, g_loss


def main():	
	tf.reset_default_graph() #reset all graph
	
	plt.axis('off')
	d_train, g_train = trains()
	fakes = make_fakes()
	
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	for i in range(50000):
		with tf.device('/device:GPU:0'):
			batches = mnist.train.next_batch(batch_size)[0]
			_, _d_ross, _dr, _df = sess.run([d_train], feed_dict={input_data:batches})
			_ = sess.run([g_train], feed_dict={input_data:batches})
			if i % 100== 0 :
				_fakes = sess.run([fakes], feed_dict={input_data:batches})
				plt.figure(figsize=(8,1))	
				plt.axis('off')
				plt.imshow(_fakes[0][0].astype(np.uint8))
				plt.show()