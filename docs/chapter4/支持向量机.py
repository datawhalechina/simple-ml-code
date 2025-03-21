import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

def generate_data(n_samples=100, centers=2, random_state=42):
    """
    生成随机数据
    参数:
        n_samples: 样本数量
        centers: 类别数量
        random_state: 随机种子
    返回:
        X: 特征矩阵
        y: 标签数组
    """
    try:
        print(f"生成{centers}类{n_samples}个样本的随机数据...")
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
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

def train_and_evaluate_model(X, y, kernel='linear'):
    """
    训练和评估SVM模型
    参数:
        X: 特征矩阵
        y: 标签数组
        kernel: 核函数类型
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        print(f"\n开始训练SVM模型 (kernel={kernel})...")
        start_time = time.time()
        
        model = svm.SVC(kernel=kernel)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"模型训练完成，用时: {training_time:.2f}秒")
        
        # 评估模型
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'train_score': train_score,
            'test_score': test_score,
            'training_time': training_time,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return model, metrics, X_test, y_test
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return None, None, None, None

def plot_decision_boundary(model, X, y):
    """
    绘制决策边界
    参数:
        model: 训练好的SVM模型
        X: 特征矩阵
        y: 标签数组
    """
    try:
        print("\n绘制决策边界...")
        # 创建网格点
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # 预测网格点的类别
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制结果
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='coolwarm')
        plt.title('SVM分类结果')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.colorbar(label='类别')
        plt.show()
    except Exception as e:
        print(f"绘制决策边界时出错: {str(e)}")

def plot_confusion_matrix(cm):
    """
    绘制混淆矩阵
    参数:
        cm: 混淆矩阵
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {str(e)}")

def main():
    # 生成数据
    X, y = generate_data(n_samples=200, centers=2)
    if X is None or y is None:
        return
    
    # 数据预处理
    X_scaled = preprocess_data(X)
    if X_scaled is None:
        return
    
    # 训练和评估模型
    model, metrics, X_test, y_test = train_and_evaluate_model(X_scaled, y)
    if model is None or metrics is None:
        return
    
    # 打印评估结果
    print("\n模型评估结果:")
    print(f"训练集准确率: {metrics['train_score']:.4f}")
    print(f"测试集准确率: {metrics['test_score']:.4f}")
    print(f"训练用时: {metrics['training_time']:.2f}秒")
    print("\n分类报告:")
    print(metrics['classification_report'])
    
    # 绘制混淆矩阵
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # 绘制决策边界
    plot_decision_boundary(model, X_scaled, y)

if __name__ == "__main__":
    main() 