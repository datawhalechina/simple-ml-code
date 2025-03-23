# 第3章：决策树

## 3.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

```python
import seaborn as sns
from pandas import plotting 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
```

### 代码解释

让我们详细了解每个导入的库的作用：

1. **数据可视化库**
```python
import seaborn as sns
import matplotlib.pyplot as plt
```
- seaborn：基于 matplotlib 的高级数据可视化库，能生成更漂亮的统计图表
- matplotlib.pyplot：最基础的绘图库，用于创建各种类型的图表

2. **数据处理库**
```python
from pandas import plotting
import pandas as pd
import numpy as np
```
- pandas：强大的数据处理和分析库，提供了 DataFrame 等数据结构
- numpy：用于科学计算的基础库，特别是在处理数组和矩阵运算时非常有用

3. **机器学习相关库**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
```
- DecisionTreeClassifier：用于创建决策树分类模型
- load_iris：用于加载经典的鸢尾花数据集
- train_test_split：用于将数据集分割为训练集和测试集
- tree：提供决策树相关的工具，如可视化决策树

> **Python 库导入的三种方式**
>
> 1. **直接导入整个库**
>    ```python
>    import pandas
>    # 使用：pandas.DataFrame()
>    ```
>
> 2. **导入并使用别名**
>    ```python
>    import pandas as pd
>    # 使用：pd.DataFrame()
>    ```
>
> 3. **从库中导入特定函数**
>    ```python
>    from sklearn.tree import DecisionTreeClassifier
>    # 使用：DecisionTreeClassifier()
>    ```
>
> 不推荐使用 `import *` 的方式导入所有内容，因为这可能导致命名冲突，并且使代码的依赖关系不够明确。

## 3.2 加载和准备数据

```python
# 加载数据
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Species'] = data.target

# 查看数据信息
print(f"数据集信息：\n{df.info()}")
print(f"前五条数据：\n{df.head()}")
print(f"数据统计摘要：\n{df.describe()}")
```

### 代码解释

1. **加载数据集**
```python
data = load_iris()
```
加载著名的鸢尾花数据集，这是机器学习中经典的数据集之一。

2. **创建 DataFrame**
```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```
将数据转换为 pandas 的 DataFrame 格式：
- data.data：包含鸢尾花的测量数据
- data.feature_names：特征名称（如花瓣长度、宽度等）

3. **添加目标变量**
```python
df['Species'] = data.target
```
将鸢尾花的品种标签添加为新的列 'Species'。

4. **数据探索**
```python
print(f"数据集信息：\n{df.info()}")
print(f"前五条数据：\n{df.head()}")
```
这些命令帮助我们了解数据的基本情况：
- df.info()：显示数据类型和缺失值信息
- df.head()：显示前 5 行数据
- df.describe()：显示数据的统计摘要

## 3.3 数据可视化

```python
# 定义颜色方案
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']

# 创建小提琴图
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)

# 绘制四个特征的小提琴图
sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
plt.show()

# 创建点图
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
sns.despine(left=True)

# 绘制四个特征的点图
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
plt.show()

# 绘制 Andrews 曲线
plt.subplots(figsize=(8, 6))
plotting.andrews_curves(df, 'Species', colormap='cool')
plt.show()
```

### 代码解释

1. **定义颜色方案**
```python
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
```
创建一个颜色调色板，用于保持可视化的一致性和美观性。

2. **小提琴图**
```python
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)
```
- 创建 2×2 的子图网格
- sns.despine()：移除上方和右方的坐标轴边框，使图表更简洁
- violinplot 显示了每个特征在不同品种间的分布情况

3. **点图**
```python
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
```
点图显示了每个特征的平均值在不同品种间的变化趋势。

4. **Andrews 曲线**
```python
plotting.andrews_curves(df, 'Species', colormap='cool')
```
Andrews 曲线是一种多维数据可视化方法，可以帮助我们发现数据中的模式和聚类。

## 3.4 训练决策树模型

```python
# 准备数据
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

X = df.drop(columns="Species")
y = df["Species"]
feature_names = X.columns
labels = y.unique()

# 分割数据集
X_train, test_x, y_train, test_lab = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建和训练模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 输出决策树结构
text_representation = tree.export_text(model)
print(text_representation)

# 可视化决策树
plt.figure(figsize=(30, 10), facecolor='g')
tree.plot_tree(model, feature_names=feature_names, class_names=labels, 
               rounded=True, filled=True, fontsize=14)
plt.show()
```

### 代码解释

1. **数据准备**
```python
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)
```
将数字标签转换为实际的鸢尾花品种名称。

2. **特征和标签分离**
```python
X = df.drop(columns="Species")
y = df["Species"]
```
- X：包含所有特征的数据
- y：包含目标变量（鸢尾花品种）

3. **数据集分割**
```python
X_train, test_x, y_train, test_lab = train_test_split(X, y, test_size=0.4, random_state=42)
```
将数据分为训练集（60%）和测试集（40%）。

4. **创建和训练模型**
```python
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
```
- max_depth=3：限制树的深度为 3，防止过拟合
- random_state=42：设置随机种子，确保结果可重现

5. **可视化决策树**
```python
tree.plot_tree(model, feature_names=feature_names, class_names=labels, 
               rounded=True, filled=True, fontsize=14)
```
生成决策树的可视化图形，显示：
- 每个节点的分裂条件
- 每个叶节点的预测类别
- 节点中样本的分布情况

## 3.5 总结

决策树是一种直观的机器学习算法，它通过一系列问题将数据分成不同的类别。在本章中，我们：

1. 学习了如何加载和处理数据
2. 使用多种可视化方法探索数据
3. 创建并训练了一个决策树模型
4. 可视化了决策树的结构

决策树的优点是：
- 易于理解和解释
- 可以处理数值型和类别型数据
- 训练速度快
- 可以自然处理多分类问题

缺点是：
- 容易过拟合
- 对数据中的小变化敏感
- 可能产生比较复杂的树结构

在实践中，我们通常会使用决策树的改进版本，如随机森林或梯度提升树，它们可以克服单个决策树的一些缺点。 