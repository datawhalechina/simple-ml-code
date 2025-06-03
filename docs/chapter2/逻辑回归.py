import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys

# 获取项目根目录并添加到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# 从本地加载MNIST数据集
def load_mnist_data():
    from datasets.MNIST.raw.load_data import load_local_mnist
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'MNIST', 'raw')
    (X_train, y_train), (X_test, y_test) = load_local_mnist(
        x_train_path=os.path.join(base_path, 'train-images-idx3-ubyte.gz'),
        y_train_path=os.path.join(base_path, 'train-labels-idx1-ubyte.gz'),
        x_test_path=os.path.join(base_path, 't10k-images-idx3-ubyte.gz'),
        y_test_path=os.path.join(base_path, 't10k-labels-idx1-ubyte.gz'),
        normalize=True,
        one_hot=False
    )
    return X_train, y_train, X_test, y_test

# 加载数据
X_train, y_train, X_test, y_test = load_mnist_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print("Test score with L1 penalty:%.4f" % score)