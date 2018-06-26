import tensorflow as tf
import pandas as pd


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

training_epochs=10
display_step=2
batch_size=16
ckpt_path = 'ckpt/iris/iris-model.ckpt'

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

# 定义权重和偏置
w=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))


# 建立线性模型
logits=tf.matmul(x,w)+b
pre=tf.nn.softmax(logits)

# 交叉熵损失函数
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits))

#优化函数
optimizer=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)


# 计算准确率
correct_prediction=tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 初始化操作
init=tf.global_variables_initializer()

# 声明保存训练文件的Saver
saver=tf.train.Saver()
sess=tf.Session()
sess.run(init)

# 训练和测试
with tf.Session() as sess:
    sess.run(init)
    total = len(train_x)
    total_batch =total//batch_size
    for epoch in range(training_epochs):
        start = 0
        end = batch_size
        for batch in range(total_batch):
            if end >= total:
                end = total
            t_x = train_x[start:end]
            t_y = train_y[start:end]
            start += batch_size
            end += start

            _, train_cost ,train_acc= sess.run([optimizer, cost,acc], feed_dict={x: t_x, y: t_y})
        # 每训练display_step次显示结果
        if (epoch+1) % display_step == 0:
            test_cost, test_acc = sess.run([cost, acc],feed_dict={x: test_x, y: test_y})
            print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}".format(epoch+1, train_cost, train_acc,test_cost, test_acc))

    save_path = saver.save(sess, ckpt_path)
    print("Model saved in file: %s" % save_path)


# 加载模型进行重新训练
training_epochs=20
print("开始第二次训练.")
with tf.Session() as sess:
    sess.run(init)
    # 加载模型
    load_path = saver.restore(sess, ckpt_path)
    print("模型加载的文件: %s" % save_path)
    total = len(train_x)
    total_batch =total//batch_size
    for epoch in range(training_epochs):
        start = 0
        end = batch_size
        for batch in range(total_batch):
            if end >= total:
                end = total
            t_x = train_x[start:end]
            t_y = train_y[start:end]
            start += batch_size
            end += start

            _, train_cost ,train_acc= sess.run([optimizer, cost,acc], feed_dict={x: t_x, y: t_y})
        # 每训练display_step次显示结果
        if (epoch+1) % display_step == 0:
            test_cost, test_acc = sess.run([cost, acc],feed_dict={x: test_x, y: test_y})
            print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}".format(epoch + 1, train_cost,
                                                                                                train_acc, test_cost,
                                                                                                test_acc))