import tensorflow as tf

# 输出Tensorflow的版本
print(tf.__version__)

#常量
value1 =tf.constant("Hello,Tensorflow",dtype=tf.string)
value2 = tf.constant([1, 2, 3, 4, 5, 6, 7],dtype=tf.int32)
value3 = tf.constant(-1.0, shape=[2, 3],dtype=tf.float32,name="value3")
print(value1)
print(value2)
print(value3)

# 全局初始化
init=tf.global_variables_initializer()
sess=tf.Session()
# 执行初始化操作
sess.run(init)
print("value1",sess.run(value1))
print("value2",sess.run(value2))
print("value3",sess.run(value3))

# 变量
value4=tf.Variable(1,name="value4",dtype=tf.int32)
value5=tf.Variable([1,1],name="value5",dtype=tf.int32)
print("value4",value4)
print("value5",value5)

init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("value4",sess.run(value4))
print("value5",sess.run(value5))


# 改变变量value4的值
temp = value4.assign(value4 + 1)
print("temp",sess.run(temp))
print("valu4",sess.run(value4))
temp = value4.assign(value4*2)
print("temp",sess.run(temp))
print("valu4",sess.run(value4))


# 通过get_Variable创建变量
a1=tf.get_variable(name="a1",shape=[1,2],initializer=tf.truncated_normal_initializer)
b1=tf.get_variable(name="b1",shape=[2,3])
print("a1",a1)
print("b1",b1)


# 命名出冲突
name1=tf.Variable(1,name="name1",dtype=tf.int32)
name2=tf.Variable(2,name="name1",dtype=tf.int32)
print("name1:",name1)
print("name2:",name2)
# 消除注释，就会报错，说明get_variable不能命名相同的变量名称
# name3=tf.get_variable(1,name="name2",dtype=tf.int32)
# name4=tf.get_variable(2,name="name2",dtype=tf.int32)
# print("name3:",name3)
# print("name4:",name4)

# 基本运算
a1=tf.constant(1,dtype=tf.int32)
a2=tf.Variable(2,dtype=tf.int32)
# 加法
sum=tf.add(a1,a2)
# 减法
sub=tf.subtract(a2,a1)
# 乘法
mul=tf.multiply(a2,a1)
# 除法
div=tf.div(a2,a1)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print("a1 =",sess.run(a1))
print("a2 =",sess.run(a2))
print("sum =",sess.run(sum))
print("sub =",sess.run(sub))
print("mul =",sess.run(mul))
print("div =",sess.run(div))

# tf.paceholder的用法
input_1=tf.placeholder(dtype=tf.int32)
input_2=tf.placeholder(dtype=tf.int32)
out=tf.add(input_1,input_2)

print("input_1",input_1)
print("input_2",input_2)
print("out",out)
# 填充值
print("input_1 =",sess.run(input_1,feed_dict={input_1:2}))
print("input_2 =",sess.run(input_2,feed_dict={input_2:3}))
print("out =",sess.run(out,feed_dict={input_1:2,input_2:3}))



