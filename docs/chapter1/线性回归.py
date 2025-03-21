import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def true_fun(X):
    """
    真实函数，用于生成训练数据
    参数:
        X: 输入特征
    返回:
        真实值 y = 1.5x + 0.2
    """
    return 1.5*X + 0.2

def generate_data(n_samples=30, noise_level=0.05):
    """
    生成训练数据
    参数:
        n_samples: 样本数量
        noise_level: 噪声水平
    返回:
        X_train: 训练特征
        y_train: 训练标签
    """
    try:
        np.random.seed(0)  # 设置随机种子，确保结果可重现
        X_train = np.sort(np.random.rand(n_samples))
        y_train = (true_fun(X_train) + np.random.randn(n_samples) * noise_level).reshape(n_samples, 1)
        return X_train, y_train
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        return None, None

def train_and_evaluate_model(X_train, y_train):
    """
    训练模型并评估性能
    参数:
        X_train: 训练特征
        y_train: 训练标签
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 训练模型
        model = LinearRegression()
        model.fit(X_train[:, np.newaxis], y_train)
        
        # 计算评估指标
        y_pred = model.predict(X_train[:, np.newaxis])
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        
        metrics = {
            'r2_score': r2,
            'mse': mse,
            'coefficient': model.coef_[0][0],
            'intercept': model.intercept_[0]
        }
        
        return model, metrics
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return None, None

def plot_results(X_train, y_train, model, true_fun):
    """
    绘制结果
    参数:
        X_train: 训练特征
        y_train: 训练标签
        model: 训练好的模型
        true_fun: 真实函数
    """
    try:
        plt.figure(figsize=(10, 6))
        X_test = np.linspace(0, 1, 100)
        
        # 绘制预测线和真实函数
        plt.plot(X_test, model.predict(X_test[:, np.newaxis]), 
                label="预测模型", color='red', linestyle='--')
        plt.plot(X_test, true_fun(X_test), 
                label="真实函数", color='blue', linestyle='-')
        
        # 绘制训练数据点
        plt.scatter(X_train, y_train, 
                   label="训练数据", color='green', alpha=0.5)
        
        plt.xlabel("特征 X")
        plt.ylabel("目标值 y")
        plt.title("线性回归模型拟合结果")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"绘制结果时出错: {str(e)}")

def main():
    # 生成数据
    X_train, y_train = generate_data()
    if X_train is None or y_train is None:
        return
    
    # 训练和评估模型
    model, metrics = train_and_evaluate_model(X_train, y_train)
    if model is None or metrics is None:
        return
    
    # 打印结果
    print("\n模型评估结果:")
    print(f"R² 分数: {metrics['r2_score']:.4f}")
    print(f"均方误差: {metrics['mse']:.4f}")
    print(f"模型参数:")
    print(f"  斜率 (w): {metrics['coefficient']:.4f}")
    print(f"  截距 (b): {metrics['intercept']:.4f}")
    
    # 绘制结果
    plot_results(X_train, y_train, model, true_fun)

if __name__ == "__main__":
    main()