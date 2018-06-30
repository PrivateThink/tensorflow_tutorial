import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups


# 生成单词的索引
def word2index(vocab):
    word2index={}
    for i,word in enumerate(vocab):
        word2index[word.lower()]=i
    return word2index


# 将文本转成向量（这里是one-hot）
def text2vector(text,word2index,total_words):
    vector=np.zeros(total_words,dtype=float)
    for word in text.split(" "):
        vector[word2index[word.lower()]]+=1
    return vector

# 将标签转成向量
def category2vector(category):
    y=np.zeros((n_class),dtype=float)
    if category==0:
        y[0]=1
    elif category==1:
        y[1]=1
    else:
        y[2]=1

    return y


# 生成batchsize的数据集
def get_batch(df,i,batch_size,word2index,total_words):

    data=[]
    lables=[]
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    catagories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        vector=text2vector(text,word2index,total_words)
        data.append(vector)

    for catagory in catagories:
        c=category2vector(catagory)
        lables.append(c)

    return np.array(data),np.array(lables)


# 创建多层感知机
def multilayer_perceptron(x_input,weights,biases):

    layer_1=tf.matmul(x_input,weights["h1"])+biases["b1"]
    layer_1=tf.nn.relu(layer_1)

    layer_2=tf.matmul(layer_1,weights["h2"])+biases["b2"]
    layer_2=tf.nn.relu(layer_2)

    out=tf.matmul(layer_2,weights["out"])+biases["out"]

    return out


# 参数
lr=0.001
enpochs=20
batch_size=128
display_step=1

# 多层感机网络参数
n_hidden_1=128 # 第一层隐藏层的单元数
n_hidden_2=256 # 第二层隐藏层的单元数
n_class=3

categories = ["comp.graphics","sci.space","rec.sport.baseball"]
# 训练数据集
train=fetch_20newsgroups(subset="train",categories=categories)
# 测试训练集
test=fetch_20newsgroups(subset="test",categories=categories)

# 获取全部数据
# train=fetch_20newsgroups(subset="train")
#test=fetch_20newsgroups(subset="test")

print('训练集的大小:',len(train.data))
print('测试集的大小:',len(test.data))

vocab=Counter()
# 建立词典

for text in train.data:
    for word in text.split(" "):
        vocab[word.lower()]+=1

for text in test.data:
    for word in text.split(" "):
        vocab[word.lower()]+=1

total_words =len(vocab)
n_input=total_words
print("词典的大小：",total_words)
# 生成单词索引
word2index=word2index(vocab)
print("单词‘my’的索引:",word2index['my'])


# 输入占位符
x_input=tf.placeholder(tf.float32,[None,n_input],name="input")
y_output=tf.placeholder(tf.float32,[None,n_class],name="output")

#权重信息和偏置
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_class]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_class]))
}

# 创建模型
prediction =multilayer_perceptron(x_input,weights,biases)

# 定义loss和优化器
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y_output))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for e in range(enpochs):
        avg_cost=0
        total_batchs=int(len(train.data)/batch_size)

        for i in range(total_batchs):
            x,y=get_batch(train,i,batch_size,word2index,total_words)
            cost,_=sess.run([loss,optimizer],feed_dict={x_input:x,y_output:y})
            avg_cost += cost / total_batchs

        if e % display_step==0:
            print("Epoch:", '%04d' % (e + 1), "loss=", "{:.9f}".format(avg_cost))



    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_output, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(test.target)
    batch_x_test, batch_y_test = get_batch(test, 0, total_test_data,word2index,total_words)
    print("Accuracy:", accuracy.eval({x_input: batch_x_test, y_output:batch_y_test}))

    # 保存训练的模型
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)