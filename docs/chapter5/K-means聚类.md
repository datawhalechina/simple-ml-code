# K-means聚类

## 代码块
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
## 逐行解释
    import numpy as np
这行代码导入了numpy库，别名为np。numpy是一个强大的数学计算库，就像是我们手中的计算器，帮助我们进行各种数值运算。

    import matplotlib.pyplot as plt
matplotlib是一个用于绘制图表和数据可视化的库，pyplot是matplotlib的一个模块，通常用于绘制各种类型的图表。这就像是我们手中的画笔，帮助我们画出漂亮的图形。

    from sklearn.datasets import make_blobs
从sklearn库的datasets模块导入make_blobs函数，这个函数可以帮助我们生成用于聚类的模拟数据。这就像是我们自己创造一些数据来测试我们的模型。

    from sklearn.cluster import KMeans
从sklearn库的cluster模块导入KMeans类，这是一个经典的聚类算法。它就像是一个聪明的分类器，能够自动将相似的数据点分到一组。

    from sklearn.preprocessing import StandardScaler
从sklearn库的preprocessing模块导入StandardScaler类，这个类用于数据的标准化处理。这就像是我们把不同尺度的数据统一到一个标准尺度，让模型更容易学习。

## 生成数据
    # 生成随机数据
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)
## 逐行解释
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)
这行代码使用make_blobs函数生成了300个样本点，分为4个簇（centers=4）。cluster_std=2表示每个簇的标准差为2，这决定了簇的分散程度。random_state=42确保每次运行代码时生成的数据都是一样的。

## 数据标准化
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
## 逐行解释
    scaler = StandardScaler()
创建了一个StandardScaler对象，这个对象可以帮助我们将数据标准化。标准化就是将数据转换到均值为0，标准差为1的分布。

    X_scaled = scaler.fit_transform(X)
使用scaler对象对数据进行标准化处理。fit_transform方法会先计算数据的均值和标准差，然后将数据转换为标准正态分布。

## 训练模型
    # 创建并训练K-means模型
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
## 逐行解释
    kmeans = KMeans(n_clusters=4, random_state=42)
创建了一个K-means聚类器，设置聚类数为4。random_state=42确保每次运行代码时得到相同的聚类结果。

    kmeans.fit(X_scaled)
使用标准化后的数据训练K-means模型。这就像是让模型学习如何最好地将数据点分组。

## 可视化结果
    # 绘制原始数据
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    plt.title('原始数据')
## 逐行解释
    plt.figure(figsize=(12, 5))
创建一个新的图形，设置大小为12x5英寸。

    plt.subplot(1, 2, 1)
将图形分成1行2列，当前绘制第1个子图。

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
绘制原始数据点，c=y表示根据真实标签设置不同的颜色，cmap='viridis'设置颜色映射。

    plt.title('原始数据')
设置子图的标题。

    # 绘制聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='r', label='聚类中心')
    plt.title('K-means聚类结果')
    plt.legend()
    plt.show()
## 逐行解释
    plt.subplot(1, 2, 2)
绘制第2个子图。

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
绘制聚类后的数据点，c=kmeans.labels_表示根据聚类标签设置不同的颜色。

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='r', label='聚类中心')
绘制聚类中心点，使用红色'x'标记，s=200设置点的大小，linewidths=3设置线条宽度。

    plt.title('K-means聚类结果')
设置子图的标题。

    plt.legend()
添加图例。

    plt.show()
显示图形。

## 评估聚类效果
    # 计算轮廓系数
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"轮廓系数: {score:.3f}")
## 逐行解释
    from sklearn.metrics import silhouette_score
从sklearn库的metrics模块导入silhouette_score函数，这个函数用于计算轮廓系数，评估聚类效果。

    score = silhouette_score(X_scaled, kmeans.labels_)
计算轮廓系数，这个值越接近1，表示聚类效果越好。

    print(f"轮廓系数: {score:.3f}")
打印轮廓系数，保留3位小数。

## 总结
K-means聚类是一个简单但强大的无监督学习算法，它能够自动将相似的数据点分到一组。在这个例子中，我们使用了4个簇，并且通过可视化结果可以看到，算法成功地将数据点分成了4个不同的组。

K-means算法的工作原理是：
1. 随机选择k个点作为初始聚类中心
2. 将每个数据点分配到最近的聚类中心
3. 重新计算每个簇的中心点
4. 重复步骤2和3，直到聚类中心不再变化

我们使用轮廓系数来评估聚类效果，这个指标反映了簇内紧密度和簇间分离度的平衡。在实际应用中，我们还可以通过调整簇的数量（k值）来获得更好的聚类效果。 