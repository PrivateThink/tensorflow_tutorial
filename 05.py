import tensorflow as tf
import tensorflow.contrib.eager as tfe
# 导入手写体数据
from tensorflow.examples.tutorials.mnist import input_data

# 启动Eager模式，一旦启动，中途就不能中断
tfe.enable_eager_execution()

# 读取数据
mnist = input_data.read_data_sets("data/", one_hot=False)

# 参数
learning_rate = 0.1
batch_size = 128
num_steps = 1000
display_step = 100

# 数据分割器，按照batch_size大小分
# mnist.train.images 手写体的图片
# mnist.train.labels 手写体标签
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels)).batch(batch_size)
# 数据迭代器，每次迭代大小为batch_size
dataset_iter = tfe.Iterator(dataset)

# 设置权重
W = tfe.Variable(tf.zeros([784, 10]), name='weights')
b = tfe.Variable(tf.zeros([10]), name='bias')


# 模型
def logistic_regression(inputs):
    return tf.matmul(inputs, W) + b


# 交叉熵损失函数
def loss_fn(inference_fn, inputs, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))


# 计算准确率
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 随机梯度优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 计算梯度
grad = tfe.implicit_gradients(loss_fn)

# 训练
average_loss = 0.
average_acc = 0.
for step in range(num_steps):

    # 迭代数据
    try:
        d = dataset_iter.next()
    except StopIteration:
        dataset_iter = tfe.Iterator(dataset)
        d = dataset_iter.next()

    x_batch = d[0]
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # 计算每批次的losdd
    batch_loss = loss_fn(logistic_regression, x_batch, y_batch)
    # 平均 loss
    average_loss += batch_loss
    # 计算每批次的准确率
    batch_accuracy = accuracy_fn(logistic_regression, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        print("Initial loss= {:.9f}".format(average_loss))

    # 优化器优化
    optimizer.apply_gradients(grad(logistic_regression, x_batch, y_batch))

    # 显示
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

# 验证测试集
testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(logistic_regression, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))