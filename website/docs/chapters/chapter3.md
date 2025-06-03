# 决策树
## 导入必要的库

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
## 库函数说明

经过前两章的学习，同学们一定已经知道了这些库的基本作用。让我们详细了解每个新引入的库的功能：

- **seaborn**: 基于matplotlib的数据可视化库，能生成美观的统计图表
- **pandas**: 强大的数据处理和分析库
- **numpy**: 用于处理多维数组和进行数学计算的基础库
- **matplotlib**: 经典的绘图库，其中pyplot模块提供了类似MATLAB的绘图接口
- **sklearn**相关组件：
  - `tree.DecisionTreeClassifier`: 用于创建决策树分类模型
  - `datasets.load_iris`: 用于加载经典的鸢尾花数据集
  - `model_selection.train_test_split`: 用于划分训练集和测试集
  - `tree`: 提供决策树相关的工具，包括可视化功能

### Python导入库的三种方式

#### 1. 直接导入
```python
import pandas
```
使用时需要通过完整路径访问：
```python
pandas.DataFrame(变量)
```

这种方式清晰地显示了函数来源，但写起来较为冗长。我们也可以使用别名来简化代码：
```python
import pandas as pd  # pd是pandas的常用别名
```

#### 2. 从模块导入特定函数
```python
from sklearn import tree
```
这种方式直接导入所需的函数，使用时不需要写完整路径：
```python
tree()
```

#### 3. 导入所有内容（不推荐）
```python
from pandas import *
```
这种方式会导入模块的所有内容，可能会造成命名冲突，不利于代码维护。

### 扩展知识

库函数是Python生态系统的重要组成部分：

- **本质**: 其他开发者编写并分享的代码集合
- **获取方式**: 
  - 部分基础库随Python安装
  - 第三方库需要通过pip等工具安装
- **意义**: 
  - 避免重复造轮子
  - 利用社区智慧
  - 提高开发效率

我们鼓励同学们：
1. 探索更多有用的库
2. 学习库的最佳实践
3. 在实践中灵活运用

## 数据加载与预处理

```python
data = load_iris()
```
这里我们使用导入的load_iris函数，将加载的数据集存入data变量中  

```python
# 将数据转换为DataFrame格式
df = pd.DataFrame(data.data, columns=data.feature_names)

# 添加品种标签
df['Species'] = data.target

# 查看数据集信息
print(f"数据集信息：\n{df.info()}")

# 查看前五条数据
print(f"前五条数据：\n{df.head()}") 

# 查看统计摘要
df.describe()
```

这段代码完成了以下操作：
- 将数据集转换为pandas的DataFrame格式，使数据更易操作
- 添加鸢尾花品种标签作为新的列'Species'
- 输出数据集的基本信息，包括数据类型和缺失值
- 查看数据的前5行，用于检查数据结构
- 生成统计摘要，包括均值、标准差、最小值、最大值等

## 数据可视化

### 1. 小提琴图

```python
# 定义颜色调色板
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']

# 创建2x2的子图
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)  # 删除上方和右方坐标轴边框

# 绘制四个特征的小提琴图
sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
plt.show()
```
绘制四个小提琴图（violinplot）。每个图显示了每个特征在不同鸢尾花品种上的分布情况，sns.violinplot是用于绘制小提琴图的函数  

### 2. 点图

```python
# 创建2x2的子图
f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
sns.despine(left=True)

# 绘制四个特征的点图
sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
plt.show()
```

每个点图显示了特征的平均值在不同鸢尾花品种之间的变化。

### 3. 安德鲁曲线

```python
# 绘制安德鲁曲线
plt.subplots(figsize=(8, 6))
plotting.andrews_curves(df, 'Species', colormap='cool')
plt.show()
```
使用andrew_curves绘制安德鲁曲线，是一种多维数据可视化的方法，可用于展示数据中不同类别的分布。

## 模型训练与可视化

### 1. 数据准备

```python
# 转换标签
target = np.unique(data.target)
target_names = np.unique(data.target_names)
targets = dict(zip(target, target_names))
df['Species'] = df['Species'].replace(targets)

# 分离特征和标签
X = df.drop(columns="Species")
y = df["Species"]
feature_names = X.columns
labels = y.unique()

# 划分训练集和测试集
X_train, test_x, y_train, test_lab = train_test_split(X, y, test_size=0.4, random_state=42)
```

### 2. 模型训练

```python
# 创建并训练模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 输出决策树的文本表示
text_representation = tree.export_text(model)
print(text_representation)
```

### 3. 决策树可视化

```python
# 绘制决策树
plt.figure(figsize=(30, 10), facecolor='g')
tree.plot_tree(model, feature_names=feature_names, class_names=labels, 
               rounded=True, filled=True, fontsize=14)
```

## 总结

### 主要步骤

本章我们学习了如何使用决策树来分类鸢尾花数据集，主要包括以下步骤：

1. **数据准备**
   - 导入必要的库
   - 加载鸢尾花数据集
   - 数据预处理和特征提取

2. **数据可视化**
   - 使用小提琴图展示特征分布
   - 使用点图显示特征均值变化
   - 使用安德鲁曲线展示多维特征

3. **模型训练与评估**
   - 划分训练集和测试集
   - 训练决策树模型
   - 可视化决策树结构

### 算法特点

决策树是一种直观的机器学习算法，具有以下特点：

1. **工作原理**
   - 通过一系列问题将数据分类
   - 每个节点代表一个特征判断
   - 叶节点表示最终分类结果

2. **实际应用**
   - 本例中使用四个特征（萼片长度、宽度，花瓣长度、宽度）
   - 预测鸢尾花品种
   - 通过可视化直观展示决策过程

3. **主要优势**
   - 模型可解释性强
   - 决策过程透明
   - 易于理解和实现
   - 适合处理分类问题