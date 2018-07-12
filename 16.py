import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist=input_data.read_data_sets("data/mnist/",one_hot=True)


# 参数
lr=0.01 # 学习率
steps=30000 # 迭代次数
batch_size=256 # 批量的大小

display=1000 # 训练100次显示

# 网络参数
hidden_1=256 # 第一层隐藏层的单元数
hidden_2=128 # 第二层隐藏层的单元数
input_num=784 # 手写体的输入大小【28*28=784】


# 定义占位符
X=tf.placeholder(dtype=tf.float32,shape=[None,input_num])


# 定义权重和偏置

weights={
    'encoder_h1':tf.Variable(tf.random_normal([input_num,hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([hidden_1,hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([hidden_2,hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([hidden_1,input_num]))
}

b = {
    'encoder_b1': tf.Variable(tf.random_normal([hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([input_num])),
}


# 创建编码器
encoder_layer1=tf.matmul(X,weights["encoder_h1"])+b["encoder_b1"]
# 利用激活函数
encoder_layer1=tf.nn.sigmoid(encoder_layer1)
encoder_layer2=tf.matmul(encoder_layer1,weights["encoder_h2"])+b["encoder_b2"]
# 利用激活函数
encoder_layer2=tf.nn.sigmoid(encoder_layer2)

# 创建解码器
decoder_layer1=tf.matmul(encoder_layer2,weights["decoder_h1"])+b["decoder_b1"]
# 利用激活函数
decoder_layer1=tf.nn.sigmoid(decoder_layer1)
decoder_layer2=tf.matmul(decoder_layer1,weights["decoder_h2"])+b["decoder_b2"]
# 利用激活函数
decoder_layer2=tf.nn.sigmoid(decoder_layer2)

# 预测值
y_pred=decoder_layer2
# 真实值
y_true=X

# 定义损失函数，这里用平方差
loss=tf.reduce_mean(tf.pow(y_true-y_pred,2))
# 优化函数
optimizer=tf.train.RMSPropOptimizer(lr).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(1,steps+1):

    x,_=mnist.train.next_batch(batch_size)
    _,cost=sess.run([optimizer,loss],feed_dict={X:x})

    # 显示
    if i%display==0:
        print("Step % i: loss:%f" %(i,cost))


# 测试
# 测试图片的数量 n*n=16
n = 4
original = np.empty((28 * n, 28 * n))
reconsitution = np.empty((28 * n, 28 * n))
for i in range(n):
    # 获取测试数据
    batch_x, _ = mnist.test.next_batch(n)
    # 编码和解码测试数据
    g = sess.run(decoder_layer2, feed_dict={X: batch_x})

    # 显示原始数据
    for j in range(n):
        # 生成图片，一张图片的大小为28*28,
        original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # 显示重构图片
    for j in range(n):
        # 显示
        reconsitution[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

print("原始图片")
plt.figure(figsize=(n, n))
plt.imshow(original, origin="upper", cmap="gray")
plt.show()

print("重构图片")
plt.figure(figsize=(n, n))
plt.imshow(reconsitution, origin="upper", cmap="gray")
plt.show()








