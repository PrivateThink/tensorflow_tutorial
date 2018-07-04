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

#嵌入的维度
embedding_size=100
#卷积核的数量
num_filters=32
#卷积核的大小
filter_size=5
#句子的长度
sequence_length=60
#类别
num_classes=2
batchsize=64
epochs=10

neg_dir="data/rt-polaritydata/rt-polarity.neg"
pos_dir="data/rt-polaritydata/rt-polarity.pos"

x_text, y = load_data_and_labels(pos_dir,neg_dir)

# 处理文本，创建字典处理器
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(sequence_length)
vocab_processor.fit(x_text)
x = np.array(list(vocab_processor.transform(x_text)))
print("字典的大小：{:d}".format(len(vocab_processor.vocabulary_)))
print("x的大小", x.shape)
print("y的大小", y.shape)
vocab_size = len(vocab_processor.vocabulary_)

# 按照8:2分割训练集和测试集
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,shuffle=True)
print("训练集/测试集: {:d}/{:d}\n".format(len(x_train), len(x_test)))

# 定义占位符
x_input=tf.placeholder(tf.int32,[None,sequence_length],name="input")
y_output=tf.placeholder(tf.float32,[None,num_classes],name="y_output")
keep_prob = tf.placeholder(tf.float32)
l2_loss = tf.constant(0.0)

# 嵌入层
embedded = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
# 嵌入操作的结果是形状为[None，sequence_length，embedding_size]
embedded_chars = tf.nn.embedding_lookup(embedded, x_input)
#  TensorFlow的卷积输入是四维的，所以拓展最后一维为1，shape为[None，sequence_length，embedding_size，1]
embedding_inputs_expanded = tf.expand_dims(embedded_chars, -1)

# 卷积
filter_shape = [filter_size, embedding_size, 1, num_filters]
filter_= tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_")
# 卷积以后的结果为 shape=[]
conv = tf.nn.conv2d(embedding_inputs_expanded,filter=filter_,strides=[1,1,1,1], padding="VALID",name='conv')
b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
pooled = tf.nn.max_pool(
                    h,
                   ksize=[1, sequence_length - filter_size + 1, 1, 1],
                   strides=[1, 1, 1, 1],
                   padding='VALID',
                   name="pool")


pool_flat = tf.reshape(pooled, [-1, num_filters])
drop = tf.nn.dropout(pool_flat, keep_prob)
W = tf.get_variable(
             "W",
             shape=[num_filters, num_classes],
              initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)
scores = tf.nn.xw_plus_b(drop, W, b, name="scores")

predictions = tf.argmax(scores, 1, name="predictions")
#  损失函数
losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_output)
loss = tf.reduce_mean(losses)+0.001*l2_loss
# 准确率
correct_predictions = tf.equal(predictions, tf.argmax(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
# 优化函数
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:
    sess.run(init)
    length=len(x_train)
    total_batch=length//batchsize
    for e in range(epochs):
        ava_acc=0
        start=0;
        end=batchsize
        for batch in range(total_batch):

            if end >=length:
                end=length
            x = x_train[start:end]
            y = y_train[start:end]

            start+=batchsize
            end+=start

            _,train_loss,train_acc=sess.run([optimizer,loss,accuracy],feed_dict={x_input:x,y_output:y,keep_prob:0.75})
            ava_acc+=train_acc
        ava_acc=ava_acc/len(range(total_batch))
        print("epochs= ",e,"loss=",train_loss,"ava_acc=",ava_acc)
        test_loss,test_acc = sess.run([loss, accuracy], feed_dict={x_input: x_test, y_output: y_test, keep_prob: 1})
        print("test epochs= ", e, "loss=", test_loss, "test acc=", test_acc)

