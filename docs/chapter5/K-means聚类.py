import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time

def generate_data(n_samples=300, centers=4, cluster_std=2, random_state=42):
    """
    生成随机数据
    参数:
        n_samples: 样本数量
        centers: 聚类中心数量
        cluster_std: 聚类标准差
        random_state: 随机种子
    返回:
        X: 特征矩阵
        y: 真实标签
    """
    try:
        print(f"生成{centers}个聚类中心，{n_samples}个样本的随机数据...")
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                         cluster_std=cluster_std, random_state=random_state)
        return X, y
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        return None, None

def preprocess_data(X):
    """
    数据预处理：标准化
    参数:
        X: 特征矩阵
    返回:
        标准化后的特征矩阵
    """
    try:
        print("对数据进行标准化处理...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    except Exception as e:
        print(f"数据预处理时出错: {str(e)}")
        return None

def train_and_evaluate_model(X, n_clusters=4):
    """
    训练和评估K-means模型
    参数:
        X: 特征矩阵
        n_clusters: 聚类数量
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 训练模型
        print(f"\n开始训练K-means模型 (n_clusters={n_clusters})...")
        start_time = time.time()
        
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        
        training_time = time.time() - start_time
        print(f"模型训练完成，用时: {training_time:.2f}秒")
        
        # 计算评估指标
        silhouette = silhouette_score(X, model.labels_)
        calinski = calinski_harabasz_score(X, model.labels_)
        davies = davies_bouldin_score(X, model.labels_)
        
        metrics = {
            'training_time': training_time,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies,
            'inertia': model.inertia_,
            'n_iterations': model.n_iter_
        }
        
        return model, metrics
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return None, None

def plot_clustering_results(X, y_true, model, title_prefix=''):
    """
    绘制聚类结果
    参数:
        X: 特征矩阵
        y_true: 真实标签
        model: 训练好的K-means模型
        title_prefix: 标题前缀
    """
    try:
        print("\n绘制聚类结果...")
        plt.figure(figsize=(15, 5))
        
        # 绘制原始数据
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
        plt.title(f'{title_prefix}原始数据')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 绘制聚类结果
        plt.subplot(1, 3, 2)
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis')
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                   marker='x', s=200, linewidths=3, color='r', label='聚类中心')
        plt.title(f'{title_prefix}K-means聚类结果')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.legend()
        
        # 绘制聚类边界
        plt.subplot(1, 3, 3)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        plt.scatter(X[:, 0], X[:, 1], c=model.labels_, alpha=0.8, cmap='viridis')
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                   marker='x', s=200, linewidths=3, color='r', label='聚类中心')
        plt.title(f'{title_prefix}聚类边界')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"绘制聚类结果时出错: {str(e)}")

def print_metrics(metrics):
    """
    打印评估指标
    参数:
        metrics: 评估指标字典
    """
    try:
        print("\n聚类评估指标:")
        print(f"训练用时: {metrics['training_time']:.2f}秒")
        print(f"迭代次数: {metrics['n_iterations']}")
        print(f"轮廓系数: {metrics['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz指数: {metrics['calinski_harabasz_score']:.3f}")
        print(f"Davies-Bouldin指数: {metrics['davies_bouldin_score']:.3f}")
        print(f"惯性: {metrics['inertia']:.3f}")
    except Exception as e:
        print(f"打印评估指标时出错: {str(e)}")

def main():
    # 生成数据
    X, y = generate_data(n_samples=300, centers=4, cluster_std=2)
    if X is None or y is None:
        return
    
    # 数据预处理
    X_scaled = preprocess_data(X)
    if X_scaled is None:
        return
    
    # 训练和评估模型
    model, metrics = train_and_evaluate_model(X_scaled)
    if model is None or metrics is None:
        return
    
    # 打印评估指标
    print_metrics(metrics)
    
    # 绘制聚类结果
    plot_clustering_results(X_scaled, y, model)

if __name__ == "__main__":
    main() 