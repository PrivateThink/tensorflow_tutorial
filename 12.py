import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

# 启动Eager模式
tfe.enable_eager_execution()


# 定义网络结构
class NeuralNet(tfe.Network):

    def __init__(self):
        super(NeuralNet,self).__init__()
        # 第一层隐藏层
        self.layer1=self.track_layer(tf.layers.Dense(n_hidden_1,activation=tf.nn.relu))
        # 第二层隐藏层
        self.layer2 = self.track_layer(tf.layers.Dense(n_hidden_2, activation=tf.nn.relu))
        self.out_layer = self.track_layer(tf.layers.Dense(num_classes))

    def call(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        return self.out_layer(x)


# 定义损失函数
def loss_fn(inference_fn,inputs,labels):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs),labels=labels))


# 定义准确率
def accuracy_fn(inference_fn,inputs,labels):
    pred = tf.nn.softmax(inference_fn(inputs))
    correct = tf.equal(tf.argmax(pred, 1), labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# 参数
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# 网络参数
n_hidden_1 = 256 # 第一层隐藏层的单元数
n_hidden_2 = 256 # 第二隐藏层的单元数
num_input = 784 # 输入的手写体大小 28*28=784
num_classes = 10 # 手写体的类别

mnist = input_data.read_data_sets("data/mnist/", one_hot=False)

# 按照batch_size的大小切割数据
data=tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels)).batch(batch_size)
# 生成迭代器
iter=tfe.Iterator(data)

# 定义网络结构
neural_net = NeuralNet()

# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 计算梯度
grad = tfe.implicit_gradients(loss_fn)

# 训练阶段

average_loss = 0.
average_acc = 0.
for step in range(num_steps):
    # 跌代训练数据
    try:
        d = iter.next()
    except StopIteration:
        iter = tfe.Iterator(data)
        d = iter.next()

    # 图片
    x_batch = d[0]
    # 标签
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # 计算每一个batch的损失函数
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    # 计算准确率
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        # 显示
        print("Initial loss= {:.9f}".format(average_loss))

    # 利用梯度更新变量
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))

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



# 验证模型
testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(neural_net, testX, testY)
print("验证集 Accuracy: {:.4f}".format(test_acc))