import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 参数
lr=0.01 # 学习率
epochs=1000 # 训练次数
display_step=50 # 每训练50次显示效果

'''
准备数据
'''
# x轴的点坐标
train_x= np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# y轴的点坐标
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples=train_x.shape[0]
print("train_x",train_x.shape)
print("train_y",train_x.shape)

'''
建立模型
'''
# 模型的输入
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

# 模型的权重
w=tf.Variable(np.random.randn(),name="weight")
# 模型的偏置
b=tf.Variable(np.random.randn(),name="bias")

# 线性模型
pred = tf.add(tf.multiply(x, w),b)

# 平方误差
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
# 利用随机梯度方法优化
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# 初始化session
init=tf.global_variables_initializer()

'''
开始训练
'''
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for (t_x,t_y) in zip(train_x,train_y):
            sess.run(optimizer, feed_dict={x: t_x, y: t_y})

        # 每迭代20次就显示结果
        if (epoch+1) % display_step==0:
            c=sess.run(cost,feed_dict={x: train_x, y:train_x})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),  "W=", sess.run(w), "b=", sess.run(b))

    print("优化完成")
    training_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
    print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

     # 画图显示
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(w) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

