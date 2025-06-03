import seaborn as sns
from pandas import plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

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
        # 绘制violinplot
        f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        sns.despine(left=True)
        sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
        sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
        sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
        sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
        plt.show()
        
        # 绘制pointplot
        f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
        sns.despine(left=True)
        sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
        sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
        sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
        sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
        plt.show()
        
        plt.subplots(figsize=(8,6))
        plotting.andrews_curves(df,'Species',colormap='cool')
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
        X_train,test_x,y_train,test_lab=train_test_split(X,y,test_size=0.4,random_state=42)
        
        model=DecisionTreeClassifier(max_depth=3,random_state=42)
        model.fit(X_train,y_train)
        
        text_representation=tree.export_text(model)
        print(text_representation)
        
        plt.figure(figsize=(30,10),facecolor='g')
        a=tree.plot_tree(model,feature_names=feature_names,class_names=labels,rounded=True,filled=True,fontsize=14)
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        return None, None, None, None



def main():
    # 加载和准备数据
    df, X, y, feature_names, labels = load_and_prepare_data()
    if df is None:
        return
    
    # 数据探索
    explore_data(df)
    
    train_and_evaluate_model(X, y, feature_names, labels)

if __name__ == "__main__":
    main()