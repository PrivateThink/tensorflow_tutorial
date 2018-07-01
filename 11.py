from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 定义网络
def neural_net(x_dict):
    # x_dict是input_fn返回的x
    x = x_dict['images']
    # 第一层隐藏层
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # 第二层隐藏层
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # 输出层
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# 定义模型相关部分
def model_fn(features, labels, mode):
    # 创建网络
    logits = neural_net(features)

    # 预测
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # 如果是预测模式，就直接返回，不用进行下面的工作
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # 损失函数和优化器
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # 验证模型的准确率
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # 构建模型的estimator
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


tf.logging.set_verbosity(tf.logging.INFO)

# 参数
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# 网络参数
n_hidden_1 = 256 # 第一层隐藏层的单元数
n_hidden_2 = 256 # 第二隐藏层的单元数
num_input = 784 # 输入的手写体大小 28*28=784
num_classes = 10 # 手写体的类别


mnist = input_data.read_data_sets("data/mnist/", one_hot=False)
# 定义输入的函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)


# 创建Estimator
model = tf.estimator.Estimator(model_fn)
# 训练模型
model.train(input_fn, steps=num_steps)

# 验证模型
# 定义验证输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# 进行验证
model.evaluate(input_fn)
