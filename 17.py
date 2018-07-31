import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets("data/", one_hot=True)

# 参数设置
epochs = 100000
batch_size = 128
lr = 0.0002
image_dim = 784
# 生成器隐藏层的大小
gen_hidden = 256
# 识别器隐藏层的大小
disc_hidden = 256
# 噪音的大小
noise_dim = 100


# 初始化权重和参数
def init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# 权重和偏置设置
weights = {
    'gen_hidden1': tf.Variable(init([noise_dim, gen_hidden])),
    'gen_out': tf.Variable(init([gen_hidden, image_dim])),
    'disc_hidden1': tf.Variable(init([image_dim, disc_hidden])),
    'disc_out': tf.Variable(init([disc_hidden, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# 输入占位符
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name="gen_input")
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name="disc_input")

# 创建生成器
gen_layer = tf.matmul(gen_input, weights['gen_hidden1']) + biases['gen_hidden1']
gen_layer = tf.nn.relu(gen_layer)
gen_out = tf.matmul(gen_layer, weights['gen_out']) + biases['gen_out']
gen_out = tf.nn.sigmoid(gen_out)

# 创建两个识别器。一个用来识别真的图片，另外一个用来识别生成器生成的图片
# 识别真实的图片
disc_layer1 = tf.matmul(disc_input, weights['disc_hidden1']) + biases['disc_hidden1']
disc_layer1 = tf.nn.relu(disc_layer1)
disc_real = tf.matmul(disc_layer1, weights['disc_out']) + biases['disc_out']
disc_real = tf.nn.sigmoid(disc_real)

# 识别生成器生成的图片
disc_layer2 = tf.matmul(gen_out, weights['disc_hidden1']) + biases['disc_hidden1']
disc_layer2 = tf.nn.relu(disc_layer2)
disc_fake = tf.matmul(disc_layer2, weights['disc_out']) + biases['disc_out']
disc_fake = tf.nn.sigmoid(disc_fake)

# 建立损失函数
# 生成器损失函数
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
# 识别器损失函数
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# 优化器
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]

disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]
gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_loss, var_list=gen_vars)
disc_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(disc_loss, var_list=disc_vars)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 开始训练
for i in range(1, epochs + 1):
    # 获取数据
    batch_x, _ = mnist.train.next_batch(batch_size)
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    feed_dict = {disc_input: batch_x, gen_input: z}
    _, _, gl, dl = sess.run([gen_optimizer, disc_optimizer, gen_loss, disc_loss],
                            feed_dict=feed_dict)
    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

#生成器生成图片
n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # 噪音输入.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # 将噪音生成图片
    g = sess.run(gen_out, feed_dict={gen_input: z})
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # 画图
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()
