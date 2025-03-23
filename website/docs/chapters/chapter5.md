# 第5章：K-means聚类

## 5.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

### 代码解释

让我们详细了解每个导入的库的作用：

1. **数学计算库**
```python
import numpy as np
```
numpy 是一个强大的数学计算库，提供了丰富的数组操作和数值计算功能。

2. **数据可视化库**
```python
import matplotlib.pyplot as plt
```
matplotlib 是一个强大的绘图库，pyplot 是其中最常用的模块，用于创建各种类型的图表。

3. **机器学习相关库**
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```
- make_blobs：用于生成用于聚类的模拟数据
- KMeans：K-means 聚类算法的实现
- StandardScaler：用于数据标准化
- silhouette_score：用于评估聚类效果的指标

> **无监督学习与聚类**
>
> K-means 是一种无监督学习算法，它的特点是：
> 1. 不需要标签数据
> 2. 自动发现数据中的模式
> 3. 将相似的数据点分组
> 4. 需要预先指定簇的数量
>
> 这就像是将一堆彩色珠子自动分类，不需要告诉算法每个珠子的颜色，它会自己找出相似的珠子并将它们分到一组。

## 5.2 生成和准备数据

```python
# 生成随机数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 代码解释

1. **生成模拟数据**
```python
X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)
```
使用 make_blobs 函数生成聚类数据：
- n_samples=300：生成 300 个样本点
- centers=4：设置 4 个簇中心
- cluster_std=2：设置簇的标准差，控制簇的分散程度
- random_state=42：设置随机种子，确保结果可重现

2. **数据标准化**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
使用 StandardScaler 对数据进行标准化：
- 创建标准化器对象
- 计算并应用标准化转换
- 使数据满足均值为 0，标准差为 1 的分布

## 5.3 创建和训练模型

```python
# 创建并训练K-means模型
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
```

### 代码解释

1. **创建模型**
```python
kmeans = KMeans(n_clusters=4, random_state=42)
```
创建 K-means 聚类器：
- n_clusters=4：指定要将数据分成 4 个簇
- random_state=42：设置随机种子，确保结果可重现

2. **训练模型**
```python
kmeans.fit(X_scaled)
```
使用标准化后的数据训练模型。K-means 算法会：
1. 随机初始化簇中心
2. 重复以下步骤直到收敛：
   - 将每个点分配给最近的簇中心
   - 更新簇中心为该簇所有点的平均位置

> **K-means 算法的工作原理**
>
> K-means 是一个迭代算法，其工作流程如下：
> 1. **初始化**：随机选择 k 个点作为初始簇中心
> 2. **分配**：将每个数据点分配到最近的簇中心
> 3. **更新**：重新计算每个簇的中心（均值）
> 4. **重复**：重复步骤 2 和 3，直到：
>    - 簇中心不再显著变化
>    - 达到最大迭代次数
>
> 这就像是在操场上分组，每个组都有一个组长（簇中心），其他同学会加入离自己最近的组长所在的组。

## 5.4 可视化聚类结果

```python
# 创建图形
plt.figure(figsize=(12, 5))

# 绘制原始数据
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
plt.title('原始数据')

# 绘制聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r', label='聚类中心')
plt.title('K-means聚类结果')
plt.legend()
plt.show()
```

### 代码解释

1. **创建可视化布局**
```python
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
```
- 创建 12×5 英寸的图形
- 将图形分为 1 行 2 列的布局

2. **绘制原始数据**
```python
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
plt.title('原始数据')
```
- 使用散点图显示原始数据点
- 根据真实标签设置颜色
- 使用 viridis 颜色映射

3. **绘制聚类结果**
```python
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r', label='聚类中心')
```
- 绘制聚类后的数据点，颜色根据聚类标签设置
- 用红色 'x' 标记显示簇中心
- 添加图例和标题

## 5.5 评估聚类效果

```python
# 计算轮廓系数
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"轮廓系数: {score:.3f}")
```

### 代码解释

**计算轮廓系数**
```python
score = silhouette_score(X_scaled, kmeans.labels_)
```
轮廓系数是一个评估聚类质量的指标：
- 取值范围在 [-1, 1] 之间
- 越接近 1 表示聚类效果越好
- 越接近 -1 表示可能存在错误的聚类

> **如何解读轮廓系数**
>
> 轮廓系数考虑两个因素：
> 1. **内聚度**：同一簇内的点之间的距离
> 2. **分离度**：不同簇之间的距离
>
> 好的聚类结果应该是：
> - 簇内的点互相靠近（高内聚）
> - 不同簇之间的点互相远离（高分离）

## 5.6 总结

K-means 聚类是一种经典的无监督学习算法，具有以下特点：

1. **优点**：
   - 简单直观，易于实现
   - 计算速度快
   - 对大数据集仍然有效
   - 容易解释和理解

2. **缺点**：
   - 需要预先指定簇的数量
   - 对初始簇中心的选择敏感
   - 对异常值敏感
   - 只能发现球形的簇

3. **应用场景**：
   - 客户分群
   - 图像分割
   - 文档聚类
   - 异常检测

在本章中，我们学习了：
1. 如何生成和预处理聚类数据
2. 如何创建和训练 K-means 模型
3. 如何可视化聚类结果
4. 如何评估聚类效果

K-means 的核心思想是通过迭代优化来最小化簇内平方和，使得：
- 同一簇内的数据点尽可能接近
- 不同簇之间的数据点尽可能远离
- 每个簇都由其中心点（质心）来代表 