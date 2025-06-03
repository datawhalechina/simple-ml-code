# 第1章：线性回归

## 1.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

```python
import numpy as np  
import matplotlib.pyplot as plt   
from sklearn.linear_model import LinearRegression  
```

### 代码解释

1. **导入 NumPy**
```python
import numpy as np
```
`import` 是用于调用库的指令，`numpy` 是一个用于科学计算的库，特别是在处理数组和矩阵运算时非常有用。`as np` 即给 numpy 起一个简称，方便我们在后面的程序中调用。就像是我们给好朋友起昵称一样，方便快速叫到他！

2. **导入 Matplotlib**
```python
import matplotlib.pyplot as plt
```
`matplotlib` 是一个用于绘制图表和数据可视化的库，`pyplot` 是 matplotlib 的一个模块，通常用于绘制各种类型的图表。可以把它想象成我们的画笔和画板，帮助我们把数据变成直观的图像。

3. **导入 Scikit-learn**
```python
from sklearn.linear_model import LinearRegression
```
`sklearn` 是一个用于机器学习的库，包含了许多常用的机器学习算法。`linear_model` 是 sklearn 中的一个模块，专门用于线性模型的实现。`LinearRegression` 用于执行线性回归分析，它是一种最基础的回归算法，用于通过拟合一条直线来预测数据。

## 1.2 定义真实函数

```python
def true_fun(X):  # 这是我们设定的真实函数，即 ground truth 的模型
    return 1.5 * X + 0.2
```

### 代码解释

1. **函数定义**
```python
def true_fun(X):
```
用 `def` 定义了一个名为 `true_fun` 的函数，`X` 是该函数的输入参数，即自变量。`true_fun` 代表的是我们假设的真实关系函数。在机器学习中，通常我们有一个真实的目标函数 y = f(X)，但我们只有带噪声的观测数据，模型的目标是通过拟合数据来接近真实函数。

2. **函数实现**
```python
return 1.5 * X + 0.2
```
这行代码是 `true_fun` 函数的实现，代表这个函数输出的值。这是一个线性函数，斜率为 1.5，截距为 0.2。也就是说，真实的函数是一个直线，斜率为 1.5，y 轴与直线的交点为 0.2。

## 1.3 生成随机数据

```python
np.random.seed(0)  # 设置随机种子
n_samples = 30     # 设置采样数据点的个数
```

### 代码解释

1. **设置随机种子**
```python
np.random.seed(0)
```
这里调用了 np 库里的 random 方法设置了随机数生成器的种子。计算机中的随机数通常是伪随机的，它们是通过某些算法生成的。为了确保每次程序运行时得到相同的随机数，我们可以使用 seed 来设置"种子"。这意味着每次运行程序时生成的随机数序列是一样的。0 是种子的值，通常选择任意整数。设置种子可以确保程序可重复性，尤其是在做调试或者需要对比结果时非常有用。就像是播种一样，同样的种子会长出同样的植物！

2. **设置样本数量**
```python
n_samples = 30
```
设置了训练数据中的样本数量，也就是我们将生成多少个数据点，这里设定的是 30 个样本点。`n_samples` 是我们设置的变量，变量是一个个小盒子，它的名字和装载的内容由我们规定，但有一定的限制：
- 变量名只能由字母，数字和下划线组成
- 不能以数字开头
- 大小写不同代表不同变量，如 Big 和 big 是两个变量
- 不能使用 Python 的保留关键字作为变量名，如 if，else 等

## 1.4 生成训练数据

```python
X_train = np.sort(np.random.rand(n_samples))
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples, 1)
```

### 代码解释

1. **生成特征数据**
```python
X_train = np.sort(np.random.rand(n_samples))
```
- `np.random.rand()` 用于生成均匀分布的随机数
- `np.sort()` 将这些随机数从小到大排序，使得训练数据 X_train 是一个从 0 到 1 排序的数组
- 这里将一个含有三十个随机数字的从 0-1 排序的数组赋值给 X_train

2. **生成目标数据**
```python
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples, 1)
```
- `true_fun(X_train)` 计算每个样本点 X_train 对应的真实函数值，即 y = 1.5*X + 0.2
- `np.random.randn(n_samples)` 生成符合标准正态分布（均值为 0，方差为 1）的随机数，用于模拟噪声
- `* 0.05` 将噪声的幅度缩小到原来的 0.05 倍，模拟一些小的随机误差
- `.reshape(n_samples, 1)` 将 y_train 转换为一个列向量，使得它的形状为 (30, 1)

注意，这里我们是模拟数据，噪声是为了使我们模拟的数据样本更贴合现实。在现实中，不会有那么标准的线性关系。数据总会有一些"杂音"，就像我们听音乐时可能会听到一些背景噪音一样。这些噪声让我们的模型更加健壮，能够适应真实世界的不完美数据。

## 1.5 训练模型

```python
model = LinearRegression()  # 定义模型
model.fit(X_train[:, np.newaxis], y_train)  # 训练模型
print("输出参数w：", model.coef_)  # 输出模型参数w
print("输出参数b：", model.intercept_)  # 输出参数b
```

### 代码解释

1. **创建模型**
```python
model = LinearRegression()
```
创建了一个 LinearRegression 类的实例，并将其赋值给变量 model。线性回归模型的目标是找到一个最合适的直线，以使得真实数据点和拟合直线之间的误差最小化。这就像是我们在找一条最佳路径，让所有的点到这条路径的距离总和最小。

2. **训练模型**
```python
model.fit(X_train[:, np.newaxis], y_train)
```
- `.fit()` 方法用于训练模型，即根据训练数据拟合线性回归模型的参数（权重 w 和截距 b）
- `X_train[:, np.newaxis]` 这个操作将 X_train 的形状从 (30,) 转换为 (30, 1)
- 它通过 `np.newaxis` 将原来一维的数组变成二维数组
- 这是因为 sklearn 的 LinearRegression 要求输入的特征数据必须是二维数组

3. **输出模型参数**
```python
print("输出参数w：", model.coef_)
print("输出参数b：", model.intercept_)
```
- `model.coef_` 是线性回归模型的属性，表示模型学习到的回归系数（即斜率 w）
- `model.intercept_` 是线性回归模型的另一个属性，表示模型学习到的截距 b
- 在我们的例子中，真实的斜率是 1.5，截距是 0.2，我们期望学习到的参数接近这些值

## 1.6 可视化结果

```python
X_test = np.linspace(0, 1, 100)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(X_train, y_train)  # 画出训练集的点
plt.legend(loc="best")
plt.show()
```

### 代码解释

1. **生成测试数据**
```python
X_test = np.linspace(0, 1, 100)
```
`np.linspace(0, 1, 100)` 生成了一个从 0 到 1 的等间距的 100 个点。这些测试点将用于绘制模型的预测曲线。

2. **绘制模型预测结果**
```python
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
```
- `model.predict()` 通过训练好的模型来预测 X_test 上的输出值
- `plt.plot()` 将预测的结果与 X_test 绘制成一条曲线，并标记为 "Model"

3. **绘制真实函数**
```python
plt.plot(X_test, true_fun(X_test), label="True function")
```
绘制真实函数 y = 1.5 * X + 0.2 的曲线，用于与模型的预测结果进行比较。

4. **绘制训练数据点**
```python
plt.scatter(X_train, y_train)
```
使用散点图显示训练数据的分布情况。

5. **添加图例并显示**
```python
plt.legend(loc="best")
plt.show()
```
- `plt.legend()` 添加图例，自动选择最佳位置
- `plt.show()` 显示完整的图形

当程序执行后，你可能会发现输出的参数与标准的 y = 1.5 * X + 0.2 有所偏差，这是正常的。线性回归的过程就是一个不断拟合的过程，基于我们的样本数量的有限，我们只能无限趋近于准确的值。就像是射箭，我们可能无法每次都正中靶心，但通过不断练习，我们的箭会越来越接近靶心！ 