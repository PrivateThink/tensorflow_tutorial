import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split


# 清洗数据
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 加载数据
def load_data_and_labels(positive_data_file, negative_data_file):
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


# 嵌入的维度
embedding_size = 100
num_hidden = 128
sequence_length = 60
# 类别
num_classes = 2
batchsize = 256
epochs = 50
display=5

neg_dir = "data/rt-polaritydata/rt-polarity.neg"
pos_dir = "data/rt-polaritydata/rt-polarity.pos"

x_text, y = load_data_and_labels(pos_dir, neg_dir)

# 处理文本，创建字典处理器
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(sequence_length)
vocab_processor.fit(x_text)
x = np.array(list(vocab_processor.transform(x_text)))
print("字典的大小：{:d}".format(len(vocab_processor.vocabulary_)))
print("x的大小", x.shape)
print("y的大小", y.shape)
vocab_size = len(vocab_processor.vocabulary_)

# 按照8:2分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
print("训练集/测试集: {:d}/{:d}\n".format(len(x_train), len(x_test)))

# 定义占位符
x_input = tf.placeholder(tf.int32, [None, sequence_length], name="input")
y_output = tf.placeholder(tf.float32, [None, num_classes], name="y_output")

# 嵌入层
embedded = tf.Variable(tf.random_uniform([vocab_size, embedding_size]), name="embedded")
# 嵌入操作的结果是形状为[None，sequence_length，embedding_size]
embedded_chars = tf.nn.embedding_lookup(embedded, x_input)

print("embedded_chars", embedded_chars)
# 定义权重
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
# 定义偏置
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# 定义LSTM
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
# lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
# 获得输出和隐藏层的最终终状态
outputs, states = tf.nn.dynamic_rnn(lstm_cell, embedded_chars, dtype=tf.float32)
print("outputs",outputs)
logits = tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']


#  损失函数
losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_output)
loss = tf.reduce_mean(losses)
# 准确率
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
# 优化函数
# learning_rate = tf.train.exponential_decay(0.1, global_step,2, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    sess.run(init)
    length = len(x_train)
    total_batch = length // batchsize
    for e in range(epochs):
        ava_acc = 0
        start = 0
        end = batchsize
        for batch in range(total_batch):

            if end >= length:
                end = length
            x_in = x_train[start:end]
            y_in = y_train[start:end]
            start += batchsize
            end += start

            _, train_loss, train_acc = sess.run([optimizer, loss, accuracy], feed_dict={x_input: x_in, y_output: y_in})
            ava_acc += train_acc
        if e%display==0:
            ava_acc = ava_acc / len(range(total_batch))
            print("epochs= ", e, "loss=", train_loss, "ava_acc=", ava_acc)
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x_input: x_test, y_output: y_test})
            print("test epochs= ", e, "loss=", test_loss, "test acc=", test_acc)
