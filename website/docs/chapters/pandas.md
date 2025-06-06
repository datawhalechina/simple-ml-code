#Pandas基本操作
需要跑代码，请创建一个新的python文件，pandas_learning.py
首先我们学习导入pandas和创建DataFrame的两种方式
```python
import pandas as pd
import numpy as np

#1.我们先学习创建DataFrame的几种方式（什么是DataFrame请看上方加粗文字评论区）
#创建方式：1.从字典创造（字典是python独特的数据存储格式）
df1=pd.DataFrame({
    'name':['jun','wang','li'],
    'age':[25,30,222],
    'payment':[3000,9000,100000]
    }]#字典元素可以用中文，但这里为了方便学习，使用英文
print(df1)#打印字典df1
#2.从列表创建,列表是Python内置的数据类型，
#用于存储一系列有序的元素。列表非常灵活，
#可以包含不同类型的元素，如整数、浮点数、字符串等
data=[
    ['jun',25,3000],
    ['wang',30,9000],
    ['li',35,100000]
]
df2=pd.DataFrame(data,cloumns=['name','age','payment'])
#给列表中的每一列加上列名，最终会体现在表中
#其它的创建方式这里不做介绍，感兴趣的同学请自行学习，或者在评论区进行补充
```
接下来我们学习pandas的基本操作：
1.
info() 方法提供了DataFrame的基本信息，包括以下内容：
- 数据集的非空值数量（非NaN值的数量）。
- 每个列的数据类型。
- 每个列的非空值数量。
- DataFrame的内存使用情况。
2.
describe() 方法提供了DataFrame中数值列的统计摘要，包括以下内容：
- 计数（非NaN值的数量）。
- 平均值。
- 标准差。
- 最小值。
- 25%分位数（第一四分位数）。
- 中位数（50%分位数）。
- 75%分位数（第三四分位数）。
- 最大值。
```python
print(df1.info())#查看数据的基本信息
 
print(df1.describe())#查看基本统计

print(df1.head(2))#查看前2行的数据
```
数据选择操作：
```python
print(df1['name'])#选择单列

print(df1['name','age'])#选择多列

print(df1[df1['age']>30])#按条件筛选，这里是筛选age大于30的数据
```
数据修改操作：
```python
#添加新列
df1['奖金']=[1000,2000,3000]#添加奖金列及其数据
print(df1)#打印查看是否添加成功

#修改数据
df1.loc[0,'payment']=9000#把jun的工资提高到9000了
print(df1)
数据排序：
#按年龄升序排序
print(df1.short_values('age'))

#按工资降序排序
print(df1.short_values('payment',ascending=False)
#ascending=False通常在数据处理或排序的上下文中使用，
#表示在进行排序操作时，数据应该按照降序排列，
#而不是默认的升序排列。也就是说，列表或数据集中的元素会被从大到小排列，
#最大的元素会排在最前面，最小的元素会排在最后面。
```
在此我们不提供代码截图，希望各位小伙伴们自己动手敲一遍代码，亲手跑出来的更有成就感，也更能增加学习的欲望，