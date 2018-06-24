import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 读取数据
mnist = input_data.read_data_sets("data/",one_hot=True)

# 训练集和测试集分别取5000和500
train_x,train_y=mnist.train.next_batch(5000)
test_x,test_y=mnist.test.next_batch(500)

# 输入占位符
xtr=tf.placeholder(dtype=tf.float32,shape=[None,784])
xte=tf.placeholder(dtype=tf.float32,shape=[784])

# 计算距离
distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),axis=1)
# 取最近的距离
pred=tf.argmin(distance,0)
#准确率
acc=0

# 初始化
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    #循环测试训练集
    for i in range(len(test_x)):
        nn_index=sess.run(pred,feed_dict={xtr:train_x,xte:test_x[i,:]})

        print("测试 ",i,"预测类别：",np.argmax(train_y[nn_index]),"真实的类别：",np.argmax(test_y[i]))

        if np.argmax(train_y[nn_index]) == np.argmax(test_y[i]):
            acc += 1. / len(test_x)

    print("acc",acc)


