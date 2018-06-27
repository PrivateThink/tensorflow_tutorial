import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)

#参数

lr=0.001
epochs=25
batch_size=32
display_step=1
log="log/mnist/"

# 占位符
with tf.name_scope('Net_input'):
    # 手写体图片的shape为28*28=784
    x=tf.placeholder(tf.float32,[None,784],name="input")

with tf.name_scope('Net_output'):
    # 手写体图片的类别为0-9 十个类别
    y=tf.placeholder(tf.float32,[None,10],name="output")


# 权重和偏置
with tf.name_scope('Net_w'):
    w=tf.Variable(tf.zeros([784,10]),name="w")
with tf.name_scope('Net_b'):
    b=tf.Variable(tf.zeros([10]),name="b")

# 定义线性模型
with tf.name_scope('Net_logits'):
    logits=tf.matmul(x,w)+b

pred = tf.nn.softmax(logits)


with tf.name_scope('Loss'):
    # 交叉熵损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

with tf.name_scope("SGD"):
    optimizer=tf.train.GradientDescentOptimizer(lr)
    # 计算每个variable的梯度
    grads=tf.gradients(cost,tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # #根据梯度更新variable的值
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

# 准确率
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#  初始化
init=tf.global_variables_initializer()

#
tf.summary.scalar("loss",cost)
tf.summary.scalar("accuracy",accuracy)
# 遍历训练参数，将参数变化的直方图画出来
for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

# 遍历梯度，并用直方图画出来
for grad,var in grads:
    tf.summary.histogram(var.name+"/gradient",grad)

# 将所有的summary合并成图
merged_summary_op=tf.summary.merge_all()


# 开始训练
with tf.Session() as sess:

    # 执行初始化操作
    sess.run(init)

    # 将图信息写入文件log，用于Tensorboard进行展示
    summary_writer = tf.summary.FileWriter(log,graph=tf.get_default_graph())

    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 产生批次数据，用于训练
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # merged_summary_op也被用来训练
            _, c, summary = sess.run([apply_grads, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # 将训练好的summary放进文件了
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # 计算平均的loss
            avg_cost += c / total_batch
        # 显示
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))


    # 测试
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
