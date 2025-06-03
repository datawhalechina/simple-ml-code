# 决策树
## 代码块：
    import seaborn as sns
    from pandas import plotting 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn import tree
## 逐行解释
经过前两章的学习，同学们一定已经知道了这一代码块的作用，但是在这里我们引入了一些新的库，让我们来看看这些库的作用是什么吧！  
seaborn 是一个数据可视化库，基于matplotlib，能生成漂亮的统计图表  
pandas 是用于数据处理的库
numpy 是用于处理数组和数学计算的库  
matplotlib 是一个绘图库，pyplot是它的一个模块，用于绘制图表  
sklearn.tree.DecisionTreeClassifier 用于创建决策树分类模型
sklearn.datasets.load_iris 用于加载经典的iris数据集（鸢尾花数据集）  
sklearn.model_selection.train_test_split 用于将数据集分成训练集和测试集  
sklearn.tree 提供决策树相关的工具。如可视化决策树

### 科普调入库函数的三种方式
1.import pandas  
这种形式的调用，我们在使用的时候，如果需要使用它内部的函数，是以下形式的  

    pandas.DataFrame(变量)
有同学可能会问，老师老师，怎么这里是pandas，上面的代码块中则是pd呢？  
这是因为，我们在这个程序最开头引入pandas的时候，用pd代替了pandas，而在现在所处的调入方式中，并没有将pandas用别名来命名。  
而这两种写法都是可行的，写别名也是用于简洁编程  
2.from sklearn import tree   
这种形式的调用，我们在使用时，是以下形式的：

    tree()
这里为什么不需要在tree前面加上sklearn.呢？  
因为我们在这里明确的引入了对应库中的函数，IDE就会知道我们使用的是什么函数  
3.import pandas *  
这样的写法，会引入所有pandas库中的函数进入，那么在使用是和第二种方式一样，只要你知道要调用的函数的名字就行，但是我们不推荐这种写法  
至于为什么，我们鼓励你询问AI  
### 小延申
讲了这么多导入库函数的方式，可是还是有点迷糊？  
库函数本质上是别人在云端分享出来的程序，我们可以用import来调用这些程序，世界上有很多库函数，有些经典的库函数会跟随python下载到本地，  
但是，对于一些不那么普遍的，就需要我们在终端进行下载，然后，我们就可以非常舒服地调用其他人写好的程序啦！编程发展到现在，库函数真是数不胜数，我们是站在巨人肩膀上的学习者。  
我希望，同学们经过了这次的学习，可以自行去发现一些喜欢的库函数，并运用它们来进行一些有意思的事情。

## 代码块
    data=load_iris()
这里我们使用导入的load_iris函数，将加载的数据集存入data变量中  

    df=pd.DataFrame(data.data,columns=data.feature_names)
这里则是将数据集转换为pandas的DataFrame格式，使得数据更易操作，  
data.data是数据，data.feature_names是特征名称  

    df['Species']=data.target
将鸢尾花的品种标签（即分类标签）添加为新的列Species  
    
    print(f"数据集信息：\n{df.info()}")
输出数据集的基本信息，如每列的数据类型和缺失值  

    print(f"前五条数据：\n{df.head()}") 
查看数据集的前5行，通常用于检查数据的结构和内容  

    df.describe()
输出数据集的统计摘要信息（如均值、标准差、最小值、最大值等）

## 可视化部分
    antV=['#1890FF','#2FC25B','#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']
定义一个颜色调色板，用于绘画时统一颜色风格  

    # 绘制violinplot
    f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sns.despine(left=True) 
删除上方和右方坐标轴上不需要的边框，这在matplotlib中是无法通过参数实现的  

    sns.violinplot(x='Species', y=df.columns[0], data=df, palette=antV, ax=axes[0, 0])
    sns.violinplot(x='Species', y=df.columns[1], data=df, palette=antV, ax=axes[0, 1])
    sns.violinplot(x='Species', y=df.columns[2], data=df, palette=antV, ax=axes[1, 0])
    sns.violinplot(x='Species', y=df.columns[3], data=df, palette=antV, ax=axes[1, 1])
    plt.show()
绘制四个小提琴图（violinplot）。每个图显示了每个特征在不同鸢尾花品种上的分布情况，sns.violinplot是用于绘制小提琴图的函数  

    # 绘制pointplot
    f, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    sns.despine(left=True)
    sns.pointplot(x='Species', y=df.columns[0], data=df, color=antV[1], ax=axes[0, 0])
    sns.pointplot(x='Species', y=df.columns[1], data=df, color=antV[1], ax=axes[0, 1])
    sns.pointplot(x='Species', y=df.columns[2], data=df, color=antV[1], ax=axes[1, 0])
    sns.pointplot(x='Species', y=df.columns[3], data=df, color=antV[1], ax=axes[1, 1])
    plt.show()
绘制四个点图（pointplot）。每个图显示了每个特征的平均值在不同鸢尾花品种之间的变化  

    plt.subplots(figsize=(8,6))
    plotting.andrews_curves(df,'Species',colormap='cool')
    plt.show()
使用andrew_curves绘制安德鲁曲线，是一种多维数据可视化的方法，可用于展示数据中不同类别的分布。

## 准备数据并训练决策树
    target=np.unique(data.target)
    target_names=np.unique(data.target_names)
    targets=dict(zip(target,target_names))
    df['Species']=df['Species'].replace(targets)
将鸢尾花品种的数字标签转换为品种标签  

    X=df.drop(columns="Species")
    y=df["Species"]
    feature_names=X.columns
    labels=y.unique()
将特征（X）和标签（y）分开。X是包含鸢尾花特征的部分，y是对应的品种标签

    X_train,test_x,y_train,test_lab=train_test_split(X,y,test_size=0.4,random_state=42)
将数据分为训练集（60%）和测试集（40%）  

    model=DecisionTreeClassifier(max_depth=3,random_state=42)
    model.fit(X_train,y_train)
创建一个决策树分类模型，限制树的深度为3，避免过拟合，然后，使用训练数据训练模型

    text_representation=tree.export_text(model)
    print(text_representation)
输出训练好的决策树的文本表示

    plt.figure(figsize=(30,10),facecolor='g')
    a=tree.plot_tree(model,feature_names=feature_names,class_names=labels,rounded=True,filled=True,fontsize=14)
绘制决策树的图像，显示每个节点的特征和分类信息

## 总结

本章我们学习了如何使用决策树来分类鸢尾花数据集。主要步骤包括：

1. 导入必要的库
2. 加载和准备数据
3. 可视化数据分布
4. 训练决策树模型
5. 可视化决策树结构

决策树是一种直观的机器学习算法，它通过一系列问题将数据分成不同的类别。在这个例子中，我们使用了鸢尾花的四个特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度）来预测鸢尾花的品种。通过可视化决策树，我们可以清楚地看到模型是如何做出决策的。每个节点代表一个特征的判断条件，叶节点代表最终的分类结果。这种可解释性是决策树算法的一大优势。