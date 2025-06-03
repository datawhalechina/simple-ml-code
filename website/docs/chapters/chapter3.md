# 决策树
## 3.1 导入必要的库

首先，我们需要导入本章节所需的 Python 库：

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
### 代码解释

经过前两章的学习，同学们一定已经知道了这一代码块的作用，但是在这里我们引入了一些新的库，让我们来看看这些库的作用是什么吧！

1.  **导入 Seaborn**
    ```python
    import seaborn as sns
    ```
    `seaborn` 是一个数据可视化库，基于 `matplotlib`，能生成漂亮的统计图表。

2.  **导入 Pandas plotting**
    ```python
    from pandas import plotting
    ```
    从 `pandas` 库中导入 `plotting` 模块，用于数据可视化。

3.  **导入 Pandas**
    ```python
    import pandas as pd
    ```
    `pandas` 是用于数据处理的库，`pd` 是其常用别名。

4.  **导入 NumPy**
    ```python
    import numpy as np
    ```
    `numpy` 是用于处理数组和数学计算的库，`np` 是其常用别名。

5.  **导入 Matplotlib pyplot**
    ```python
    import matplotlib.pyplot as plt
    ```
    `matplotlib` 是一个绘图库，`pyplot` 是它的一个模块，用于绘制图表，`plt` 是其常用别名。

6.  **导入 DecisionTreeClassifier**
    ```python
    from sklearn.tree import DecisionTreeClassifier
    ```
    `sklearn.tree.DecisionTreeClassifier` 用于创建决策树分类模型。

7.  **导入 load_iris**
    ```python
    from sklearn.datasets import load_iris
    ```
    `sklearn.datasets.load_iris` 用于加载经典的 `iris` 数据集（鸢尾花数据集）。

8.  **导入 train_test_split**
    ```python
    from sklearn.model_selection import train_test_split
    ```
    `sklearn.model_selection.train_test_split` 用于将数据集分成训练集和测试集。

9.  **导入 sklearn tree 模块**
    ```python
    from sklearn import tree
    ```
    `sklearn.tree` 提供决策树相关的工具，如可视化决策树。

### 科普：导入库函数的三种方式

1.  **`import pandas`**
    这种形式的调用，我们在使用的时候，如果需要使用它内部的函数，是以下形式的：
    ```python
    pandas.DataFrame(变量)
    ```
    有同学可能会问，老师老师，怎么这里是 `pandas`，上面的代码块中则是 `pd` 呢？
    这是因为，我们在这个程序最开头引入 `pandas` 的时候，用 `pd` 代替了 `pandas`，而在现在所处的调入方式中，并没有将 `pandas` 用别名来命名。
    而这两种写法都是可行的，写别名也是用于简洁编程。

2.  **`from sklearn import tree`**
    这种形式的调用，我们在使用时，是以下形式的：
    ```python
    tree()
    ```
    这里为什么不需要在 `tree` 前面加上 `sklearn.` 呢？
    因为我们在这里明确的引入了对应库中的函数，IDE就会知道我们使用的是什么函数。

3.  **`import pandas *`**
    这样的写法，会引入所有 `pandas` 库中的函数进入，那么在使用是和第二种方式一样，只要你知道要调用的函数的名字就行，但是我们不推荐这种写法。
    至于为什么，我们鼓励你询问 AI。

### 小延申

讲了这么多导入库函数的方式，可是还是有点迷糊？
库函数本质上是别人在云端分享出来的程序，我们可以用 `import` 来调用这些程序，世界上有很多库函数，有些经典的库函数会跟随 Python 下载到本地。
但是，对于一些不那么普遍的，就需要我们在终端进行下载，然后，我们就可以非常舒服地调用其他人写好的程序啦！编程发展到现在，库函数真是数不胜数，我们是站在巨人肩膀上的学习者。
我希望，同学们经过了这次的学习，可以自行去发现一些喜欢的库函数，并运用它们来进行一些有意思的事情。

## 数据加载和处理

```python
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 代码解释

1.  **加载鸢尾花数据集**

    `iris = load_iris()`：使用 `load_iris()` 函数加载鸢尾花数据集，并将其存储在 `iris` 变量中。
    `X = iris.data`：将数据集的特征数据（即鸢尾花的测量值）赋值给变量 `X`。
    `y = iris.target`：将数据集的目标数据（即鸢尾花的类别）赋值给变量 `y`。

2.  **将数据集拆分为训练集和测试集**

    `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)`：
    使用 `train_test_split` 函数将数据集 `X` 和 `y` 拆分为训练集和测试集。
    `test_size=0.3` 表示测试集占总数据集的 30%。
    `random_state=42` 用于确保每次运行代码时数据集的拆分方式相同，这有助于结果的复现性。


## 可视化部分

### 绘制小提琴图

```python
sns.violinplot(data=df, orient='v')
plt.title('Violin Plot of Iris Dataset Features')
plt.show()
```

### 代码解释

1.  **绘制小提琴图**

    `sns.violinplot(data=df, orient='v')`：使用 `seaborn` 库的 `violinplot` 函数绘制小提琴图。
    `data=df` 指定绘图数据为之前创建的 `DataFrame`。
    `orient='v'` 设置小提琴图的方向为垂直。
    `plt.title('Violin Plot of Iris Dataset Features')`：设置图表的标题。
    `plt.show()`：显示图表。

### 绘制点图

```python
plotting.scatter_matrix(df, c=iris.target, figsize=(10, 10), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='viridis')
plt.suptitle('Scatter Matrix of Iris Dataset Features', y=1.02)
plt.show()
```

### 代码解释

1.  **绘制点图**

    `plotting.scatter_matrix(...)`：使用 `pandas.plotting` 模块的 `scatter_matrix` 函数绘制散点矩阵图。
    `df`：指定绘图数据为 `DataFrame`。
    `c=iris.target`：根据鸢尾花的类别 `iris.target` 为散点着色。
    `figsize=(10, 10)`：设置图表的大小为 10x10 英寸。
    `marker='o'`：设置散点的标记样式为圆形。
    `hist_kwds={'bins': 20}`：设置直方图的参数，`bins=20` 表示直方图的柱子数量。
    `s=60`：设置散点的大小。
    `alpha=.8`：设置散点的透明度。
    `cmap='viridis'`：设置颜色映射。
    `plt.suptitle('Scatter Matrix of Iris Dataset Features', y=1.02)`：设置图表的总标题，`y=1.02` 调整标题的位置。
    `plt.show()`：显示图表。

### 绘制安德鲁曲线

```python
plt.figure(figsize=(10, 6))
plotting.andrews_curves(df, 'Species')
plt.title('Andrews Curves of Iris Dataset')
plt.show()
```

### 代码解释

1.  **绘制安德鲁曲线**

    `plt.figure(figsize=(10, 6))`：创建一个新的图表，并设置其大小为 10x6 英寸。
    `plotting.andrews_curves(df, 'Species')`：使用 `pandas.plotting` 模块的 `andrews_curves` 函数绘制安德鲁曲线。
    `df`：指定绘图数据为 `DataFrame`。
    `'Species'`：指定根据 `Species` 列进行分类，不同类别的曲线将以不同颜色显示。
    `plt.title('Andrews Curves of Iris Dataset')`：设置图表的标题。
    `plt.show()`：显示图表。

## 准备数据并训练决策树

```python
# 创建决策树分类器
clf = DecisionTreeClassifier()

# 在训练集上训练模型
clf.fit(X_train, y_train)
```

### 代码解释

1.  **创建决策树分类器**

    `clf = DecisionTreeClassifier()`：创建一个 `DecisionTreeClassifier` 对象，这是 Scikit-learn 库中用于决策树分类的模型。

2.  **在训练集上训练模型**

    `clf.fit(X_train, y_train)`：使用训练数据 `X_train` 和对应的标签 `y_train` 来训练决策树模型。在训练过程中，模型会学习如何从特征中预测目标变量。


## 总结

通过本章的学习，我们了解了决策树模型的基本原理和应用。我们从数据加载和处理开始，逐步进行了模型训练和评估。以下是本章的重点回顾：

1.  **数据准备**：我们学习了如何加载鸢尾花数据集，并将其拆分为训练集和测试集。
2.  **模型训练**：我们创建了一个决策树分类器，并在训练集上对其进行了训练。
3.  **模型评估**：我们使用测试集评估了模型的性能，并通过分类报告和混淆矩阵了解了模型的准确率、精确率、召回率和 F1-分数。
4.  **可视化**：我们还学习了如何将训练好的决策树模型进行可视化，这有助于我们更好地理解模型的决策过程。