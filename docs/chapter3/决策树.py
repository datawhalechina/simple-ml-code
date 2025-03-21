import seaborn as sns
from pandas import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
import time

def load_and_prepare_data():
    """
    加载和准备鸢尾花数据集
    返回:
        df: 包含特征和标签的DataFrame
        X: 特征矩阵
        y: 标签数组
        feature_names: 特征名称列表
        labels: 标签名称列表
    """
    try:
        # 加载数据集
        data = load_iris()
        
        # 转换为DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['Species'] = data.target
        
        # 用数值替代品种名作为标签
        target = np.unique(data.target)
        target_names = np.unique(data.target_names)
        targets = dict(zip(target, target_names))
        df['Species'] = df['Species'].replace(targets)
        
        # 提取特征和标签
        X = df.drop(columns="Species")
        y = df["Species"]
        feature_names = X.columns
        labels = y.unique()
        
        return df, X, y, feature_names, labels
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None, None, None, None, None

def explore_data(df):
    """
    数据探索和可视化
    参数:
        df: 包含特征和标签的DataFrame
    """
    try:
        # 打印数据集信息
        print("\n数据集信息：")
        print(df.info())
        print("\n前5条数据：")
        print(df.head())
        print("\n各特征列的摘要信息：")
        print(df.describe())
        
        # 设置颜色主题
        antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
        
        # 绘制violinplot
        print("\n绘制特征分布图...")
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        sns.despine(left=True)
        for i, col in enumerate(df.columns[:-1]):
            sns.violinplot(x='Species', y=col, data=df, palette=antV, ax=axes[i//2, i%2])
        plt.suptitle('各特征在不同类别下的分布')
        plt.tight_layout()
        plt.show()
        
        # 绘制安德鲁曲线
        print("绘制安德鲁曲线...")
        plt.figure(figsize=(10, 6))
        plotting.andrews_curves(df, 'Species', colormap='cool')
        plt.title('鸢尾花数据集的安德鲁曲线')
        plt.show()
        
    except Exception as e:
        print(f"数据探索时出错: {str(e)}")

def train_and_evaluate_model(X, y, feature_names, labels):
    """
    训练和评估决策树模型
    参数:
        X: 特征矩阵
        y: 标签数组
        feature_names: 特征名称列表
        labels: 标签名称列表
    返回:
        model: 训练好的模型
        metrics: 评估指标
    """
    try:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        # 训练模型
        print("\n开始训练决策树模型...")
        start_time = time.time()
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
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

def visualize_tree(model, feature_names, labels):
    """
    可视化决策树
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        labels: 标签名称列表
    """
    try:
        # 以文字形式输出树
        print("\n决策树结构：")
        text_representation = tree.export_text(model)
        print(text_representation)
        
        # 绘制决策树
        print("\n绘制决策树可视化图...")
        plt.figure(figsize=(30, 10))
        tree.plot_tree(model,
                      feature_names=feature_names,
                      class_names=labels,
                      rounded=True,
                      filled=True,
                      fontsize=14)
        plt.title('决策树可视化')
        plt.show()
    except Exception as e:
        print(f"可视化决策树时出错: {str(e)}")

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
    # 加载和准备数据
    df, X, y, feature_names, labels = load_and_prepare_data()
    if df is None:
        return
    
    # 数据探索
    explore_data(df)
    
    # 训练和评估模型
    model, metrics, X_test, y_test = train_and_evaluate_model(X, y, feature_names, labels)
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
    
    # 可视化决策树
    visualize_tree(model, feature_names, labels)

if __name__ == "__main__":
    main()