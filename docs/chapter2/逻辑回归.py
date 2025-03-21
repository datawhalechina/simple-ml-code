import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

def load_mnist_data():
    """
    加载MNIST数据集
    返回:
        X_train, X_test, y_train, y_test: 训练集和测试集
    """
    try:
        print("正在加载MNIST数据集...")
        mnist = fetch_openml('mnist_784')
        X, y = mnist['data'], mnist['target']
        
        # 划分训练集和测试集
        X_train = np.array(X[:60000], dtype=float)
        y_train = np.array(y[:60000], dtype=float)
        X_test = np.array(X[60000:], dtype=float)
        y_test = np.array(y[60000:], dtype=float)
        
        print(f"数据集加载完成:")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return None, None, None, None

def preprocess_data(X_train, X_test):
    """
    数据预处理：标准化
    参数:
        X_train: 训练特征
        X_test: 测试特征
    返回:
        标准化后的训练集和测试集
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    except Exception as e:
        print(f"数据预处理时出错: {str(e)}")
        return None, None

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    训练模型并评估性能
    参数:
        X_train, X_test: 训练和测试特征
        y_train, y_test: 训练和测试标签
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 训练模型
        print("\n开始训练模型...")
        start_time = time.time()
        
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            tol=0.1,
            max_iter=1000,
            multi_class="multinomial"
        )
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
        
        return model, metrics
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return None, None

def plot_confusion_matrix(cm):
    """
    绘制混淆矩阵
    参数:
        cm: 混淆矩阵
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {str(e)}")

def plot_sample_predictions(model, X_test, y_test, n_samples=5):
    """
    绘制样本预测结果
    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        n_samples: 要显示的样本数量
    """
    try:
        # 随机选择样本
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            # 重塑图像数据
            image = X_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            pred_label = model.predict([X_test[idx]])[0]
            
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'真实: {true_label}\n预测: {pred_label}')
            plt.axis('off')
        
        plt.suptitle('样本预测结果')
        plt.show()
    except Exception as e:
        print(f"绘制样本预测结果时出错: {str(e)}")

def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_mnist_data()
    if X_train is None:
        return
    
    # 数据预处理
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    if X_train_scaled is None:
        return
    
    # 训练和评估模型
    model, metrics = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
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
    
    # 绘制样本预测结果
    plot_sample_predictions(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()