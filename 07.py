import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.factorization import KMeans


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
num_steps=50
display_step=10
k = 25

# 读取数据
def read_data(filename):
    X = []
    Y = []
    data=pd.read_csv(filename,header=0,names=CSV_COLUMN_NAMES)
    length = len(data)
    for i in range(0,length):
        x = [data.ix[i]['SepalLength'], data.ix[i]['SepalWidth'], data.ix[i]['PetalLength'], data.ix[i]['PetalWidth']]
        y = data.ix[i]['Species']
        if y==1:
            y = [1, 0, 0]
        elif y==2:
            y = [0, 1, 0]
        else:
            y=[0,0,1]
        X.append(x)
        Y.append(y)
    return X,Y


train_x,train_y=read_data("data/iris/iris_training.csv")
test_x,test_y=read_data("data/iris/iris_training.csv")

# 定义占位符
x=tf.placeholder(dtype=tf.float32,shape=[None,4])
y=tf.placeholder(dtype=tf.int32,shape=[None,3])

# 声明KMeans，k为簇群的大小
kmeans = KMeans(inputs=x, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)


# 构建Kmeans图
"""
all_scores:一个维度(num_input, num_clusters)的矩阵(或矩阵列表)，其值为输入向量和簇中心的距离
cluster_idx:与输入对应到簇群中心的id，就说分到哪一个簇群，然后输入会被分配簇群中心的id
scores:每个点到簇群中心的距离
cluster_centers_initialized：指示集群是否已初始化的标量
init_op：初始化簇群
train_op：运行迭代训练的操作
"""
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = kmeans.training_graph()
# 得到簇群的中心的id
cluster_idx = cluster_idx[0]
#平均距离
avg_distance = tf.reduce_mean(scores)
# 初始化
init_vars = tf.global_variables_initializer()


sess = tf.Session()
# 运行初始化操作
sess.run(init_vars, feed_dict={x: train_x})
sess.run(init_op, feed_dict={x: train_x})

# 训练
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={x: train_x})
    if i % 10 == 0 or i == 1:
        print("第 %i 步, 平均距离: %f" % (i, d))
print("idx",idx)
print("set(idx)",set(idx))

init_vars = tf.global_variables_initializer()

# 给每个簇群中心分配一个标签
# 计算每个簇群中心的样本个数，把样本归入离它最近的簇群中心（使用idx）
counts = np.zeros(shape=(k, 3))
for i in range(len(idx)):
    counts[idx[i]] += train_y[i]

# 将最频繁的标签分配给簇群中心
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

#测试
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Test Accuracy:", sess.run(accuracy_op, feed_dict={x: test_x, y: test_y}))

