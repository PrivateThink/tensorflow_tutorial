import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# 开启eager模式，一旦开启就不能关闭
tfe.enable_eager_execution()

# 参数
lr=0.01 # 学习率
epochs=1000 # 训练次数
display_step=50 # 每训练50次显示效果

'''
准备数据
'''
# x轴的点坐标
train_x= np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# y轴的点坐标
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples=train_x.shape[0]
print("train_x",train_x.shape)
print("train_y",train_x.shape)


# 模型的权重
w=tfe.Variable(np.random.randn(),name="weight")
# 模型的偏置
b=tfe.Variable(np.random.randn(),name="bias")


# 线性模型
def line_model(inputs):
    return inputs * w + b


# 平方差
def mean_square(model,x,y):
    return tf.reduce_sum(tf.pow(model(x)-y,2))/(2 * n_samples)


# 随机梯度下降
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
# 计算梯度
grad=tfe.implicit_gradients(mean_square)

# 计算平方差，在优化之前
cost=mean_square(line_model,train_x,train_y)

print("cost=",cost.numpy(),"w=",w.numpy(),"b",b.numpy())

# 开始训练
for step in range(epochs):

    g = grad(line_model,train_x, train_y)
    optimizer.apply_gradients(g)

    if(step+1)%display_step==0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square(line_model,train_x, train_x)),
              "W=", w.numpy(), "b=", b.numpy())

# 显示结果
plt.plot(train_x, train_y, 'ro', label='Original data')
plt.plot(train_x, np.array(w * train_x + b), label='Fitted line')
plt.legend()
plt.show()
