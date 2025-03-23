# 第2章：逻辑回归

## 2.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
```

### 代码解释

1. **导入 NumPy**
```python
import numpy as np
```
这行代码导入了 numpy 库，别名为 np。numpy 是一个强大的数学计算库，就像是我们的计算器，帮助我们进行各种数值运算。

2. **导入数据集工具**
```python
from sklearn.datasets import fetch_openml
```
从 sklearn 库的 datasets 模块导入了 fetch_openml 函数。该函数用于从 OpenML（一个开放的机器学习数据平台）加载数据集。这就像是我们去图书馆借书一样，fetch_openml 帮我们从数据仓库中取出需要的数据集。

3. **导入逻辑回归模型**
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
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
```

### 代码解释

1. **加载 MNIST 数据集**
```python
mnist = fetch_openml('mnist_784')
```
这行代码通过 fetch_openml 函数加载了名为 mnist_784 的数据集。该数据集是一个包含手写数字（0-9）的图像数据集，784 表示每个图像有 784 个像素值（28×28 像素）。mnist 是加载后的数据集对象，此后这个 mnist 就代表这个数据集，就像是我们说小明很擅长数学，那一提到小明我们就知道他数学很好。

2. **分离特征和标签**
```python
X, y = mnist['data'], mnist['target']
```
这行代码将 mnist 数据集中的数据分成特征 X 和标签 y：
- mnist['data'] 包含了图像数据（每个图像有 784 个像素）
- mnist['target'] 包含了这些图像对应的标签（即它们表示的数字，0-9）

让我们用生活中的例子来理解特征和标签：
- 特征是事物的特点，就像人的身高、体重、年龄等。在图像识别中，特征就是图像的像素值。
- 标签则是我们要预测的目标，比如根据一个人的特征判断他是学生还是老师，这里的"学生"和"老师"就是标签。

3. **准备训练数据**
```python
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
```
我们使用 mnist_784 的前 60000 个样本用于训练：
- 将 X（特征）的前 60000 个样本转换为 NumPy 数组并赋值给 X_train
- 指定数据类型为 float（浮点型），因为机器学习算法要求输入数据是浮点数类型
- 同样的操作也应用于标签 y，创建 y_train

4. **准备测试数据**
```python
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
```
将剩余的样本（从第 60001 个开始）用作测试数据：
- 转换为 NumPy 数组并赋值给 X_test 和 y_test
- 这些数据将用来测试模型的性能

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