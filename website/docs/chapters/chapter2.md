# 第2章：逻辑回归

## 2.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
```

### 代码解释

1. **导入 NumPy**
```python
import numpy as np
```
这行代码导入了 numpy 库，别名为 np。numpy 是一个强大的数学计算库，就像是我们的计算器，帮助我们进行各种数值运算。

2. **导入逻辑回归模型**
```python
from sklearn.linear_model import LogisticRegression
```
从 sklearn 库的 linear_model 模块中导入 LogisticRegression 类，这个类是一种用于分类问题的机器学习模型，常用于处理二分类或多分类问题。

> **编程语言与人类语言的相似之处**：
> 
> Python 这类的高级语言实则和人类的语言很相似，我们写程序就像是写文章一样：
> - 引入不同的库就像是引经据典
> - 给变量取名赋值，就像是给某个事物下定义
> - 程序的架构类似写文章的结构，需要交代使用了什么库、定义了什么变量、类、函数等
> - 就像文章中要交代时间地点人物，程序中也需要声明各种元素及其用法
> 
> 不要畏惧编程语言，尝试着将它与你的生活经历结合在一起，你会发现编程的乐趣！

## 2.2 加载数据集

```python
# 从本地加载MNIST数据集
def load_mnist_data():
    from datasets.MNIST.raw.load_data import load_local_mnist
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'MNIST', 'raw')
    (X_train, y_train), (X_test, y_test) = load_local_mnist(
        x_train_path=os.path.join(base_path, 'train-images-idx3-ubyte.gz'),
        y_train_path=os.path.join(base_path, 'train-labels-idx1-ubyte.gz'),
        x_test_path=os.path.join(base_path, 't10k-images-idx3-ubyte.gz'),
        y_test_path=os.path.join(base_path, 't10k-labels-idx1-ubyte.gz'),
        normalize=True,
        one_hot=False
    )
    return X_train, y_train, X_test, y_test

# 加载数据
X_train, y_train, X_test, y_test = load_mnist_data()
```

### 代码解释

1. **定义数据加载函数**
```python
def load_mnist_data():
```
这行代码定义了一个名为 load_mnist_data 的函数，用于加载本地 MNIST 数据集。函数的作用就像是一个专门的工具人，我们告诉它数据在哪里，它就帮我们把数据取出来。

2. **导入和设置路径**
```python
from datasets.MNIST.raw.load_data import load_local_mnist
base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'MNIST', 'raw')
```
这两行代码首先导入了我们自定义的 load_local_mnist 函数，然后设置了数据集的路径。base_path 就像是一个地图，告诉程序去哪里找数据文件。os.path.join 函数就像是在帮我们连接路径的各个部分，确保在不同的操作系统上都能正确找到文件。

3. **加载数据**
```python
(X_train, y_train), (X_test, y_test) = load_local_mnist(
    x_train_path=os.path.join(base_path, 'train-images-idx3-ubyte.gz'),
    y_train_path=os.path.join(base_path, 'train-labels-idx1-ubyte.gz'),
    x_test_path=os.path.join(base_path, 't10k-images-idx3-ubyte.gz'),
    y_test_path=os.path.join(base_path, 't10k-labels-idx1-ubyte.gz'),
    normalize=True,
    one_hot=False
)
```
这段代码调用 load_local_mnist 函数来加载数据。它需要四个文件路径参数：
- train-images-idx3-ubyte.gz：训练图像数据
- train-labels-idx1-ubyte.gz：训练标签数据
- t10k-images-idx3-ubyte.gz：测试图像数据
- t10k-labels-idx1-ubyte.gz：测试标签数据

normalize=True 表示我们要对图像数据进行归一化处理，将像素值从 0-255 变成 0-1 之间的小数，这样可以让模型训练更稳定。
one_hot=False 表示我们不使用独热编码来表示标签，而是直接使用 0-9 的数字标签。

4. **调用函数获取数据**
```python
X_train, y_train, X_test, y_test = load_mnist_data()
```
这行代码调用我们定义的函数来获取数据。数据集被分成了训练集（X_train, y_train）和测试集（X_test, y_test）：
- X_train：训练图像数据，包含 60000 张图片
- y_train：训练图片对应的标签
- X_test：测试图像数据，包含 10000 张图片
- y_test：测试图片对应的标签

5. **查看数据形状**
```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```
shape 是形状的意思，这些代码将打印出数据集的维度：
- X_train 的形状是 (60000, 784)，表示有 60000 个样本，每个样本有 784 个特征
- y_train 的形状是 (60000,)，表示每个样本有 1 个标签
- X_test 和 y_test 的形状与训练集类似，但样本数量不同

## 2.3 创建和训练模型

```python
clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train)
```

### 代码解释

1. **创建模型**
```python
clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
```
这行代码创建了一个逻辑回归分类模型对象 clf，使用以下参数：
- penalty="l1"：正则化参数，用于控制模型的复杂度。l1 正则化有助于产生更稀疏的模型（即很多特征的系数为 0）。这就像是给模型设置了一个"节俭"模式，让它只关注最重要的特征。
- solver="saga"：使用 saga 求解器来训练模型，这个求解器适用于大型数据集。
- tol=0.1：设置容忍度（tolerance）为 0.1，若模型的训练损失低于这个值，停止训练。

2. **训练模型**
```python
clf.fit(X_train, y_train)
```
这行代码用于调用 fit 方法来训练模型：
- 将 X_train 和 y_train 数据作为输入
- 通过训练来学习数据中图像到标签（数字）的映射关系
- 就像是老师（模型）通过学习大量例子（训练数据）来掌握知识（特征与标签的关系）

## 2.4 评估模型

```python
score = clf.score(X_test, y_test)
print("Test score with L1 penalty: %.4f" % score)
```

### 代码解释

1. **计算模型得分**
```python
score = clf.score(X_test, y_test)
```
通过 score 方法评估模型在测试数据集上的表现：
- 计算模型在 X_test 和 y_test 上的准确率
- 准确率是正确分类的样本占所有样本的比例
- 这就像是给学生（模型）一份考试（测试集），看看它能得多少分

2. **打印评估结果**
```python
print("Test score with L1 penalty: %.4f" % score)
```
打印出模型在测试集上的得分：
- 格式化为四位小数
- score 越接近 1，模型表现越好

## 2.5 总结

这个程序展示了逻辑回归在手写数字识别任务上的应用：
1. 首先加载 MNIST 数据集，并分为训练集和测试集
2. 然后创建一个逻辑回归模型并训练它
3. 最后使用训练好的模型评估在测试集上的准确性

逻辑回归虽然名字中有"回归"二字，但实际上是一种分类算法，是机器学习中的基础算法之一。它通过一个 S 形的函数（sigmoid 函数）将线性模型的输出转换为 0 到 1 之间的概率值，进而用于分类任务。就像是一个决策者，它会告诉我们："根据这些特征，我有多大把握认为这个样本属于某个类别。"