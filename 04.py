import tensorflow as tf
import matplotlib.pyplot as plt
# 导入手写体数据包
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01 #学习率
training_epochs = 25
batch_size = 100
display_step = 1
image_size=28 # 手写体图片的大小
image_flat=image_size*image_size
num_class=10 # 类别


# 展示图片
def plot_images(images):

    img_shape=(28,28)
    # 创建3*3的区域画图
    fig, axes = plt.subplots(3, 3)
    # 每个区域之间的间隔为0.3
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 画图
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        # 去除x、y轴的标题
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# 读取数据
mnist=input_data.read_data_sets("data/",one_hot=True)
# 打印训练集、测试集大小以及它们shape的大小
print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.images)))
print("- Training-shape:\t\t{}".format(mnist.train.images.shape))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Test-shape:\t\t{}".format(mnist.test.labels.shape))

# 打印标签的形式看看
print(mnist.train.labels[0:9,:])
# 展示图片
plot_images(mnist.train.images[:9])

'''
创建模型
'''
# 手写体图片的shape为 28*28=784,类别为10
# 输入占位符
x=tf.placeholder(dtype=tf.float32,shape=[None,image_flat]) # shape=[None,784]
y=tf.placeholder(dtype=tf.float32,shape=[None,num_class]) # shape=[None,10]

# 权重和偏置
w=tf.Variable(tf.zeros([image_flat,num_class]))
b=tf.Variable(tf.zeros([num_class]))

#创建模型
# 这里是线性模型
logits=tf.matmul(x, w) + b
pred = tf.nn.softmax(logits)
# 损失函数 下面二者选一
#cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
# 随机梯度下降优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 预测结果评估
'''
预测值位置（1）中最大值即分类结果，是否等于原始标签中的（1）位置。
argmax取最大值所在的下标
'''
# 准确率
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


# 训练和测试
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        # 计算总共有多少个batch_size
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            #生成batch_size条的数据提供训练
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 填充占位符
            _, train_cost ,train_acc= sess.run([optimizer, cost,accuracy], feed_dict={x: batch_xs, y: batch_ys})
        # 每训练display_step次显示结果
        if (epoch+1) % display_step == 0:
            test_cost, test_acc = sess.run([cost, accuracy],feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}".format(epoch+1, train_cost, train_acc,test_cost, test_acc))





