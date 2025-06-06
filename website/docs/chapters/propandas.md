# 进阶Pandas 
需要跑代码，请同学们创建pandas_advanced.py文件
注意，不要偷懒创建名为pandas的文件，这样会导致我们在后续的库函数使用中，因为IDE自动识别你创建的pandas而无法导入真正的pandas包
## 1.创建更复杂的数据集
```python
import pandas as pd
import numpy as np

#1.创建时间序列数据
#创建一个包含日期、风速、温度和湿度数据的DataFrame。
#首先，通过pd.date_range函数生成了一个日期范围，
#从2024年1月1日开始，持续5天，每天作为一个时间点。
dates=pd.date_range(start='2024-01-01',periods=5,freq='D')

#然后，创建了一个DataFrame，其中'日期'列使用刚才生成的日期范围，
df=pd.DataFrame({
    '日期':dates,
    '风速':np.random.normal(10,2,5),
    '温度':np.random.normal(20,5,5),
    '湿度':np.random.normal(60,10,5)
}）
#'风速'、'温度'和'湿度'列分别使用numpy库的np.random.normal函数
#生成5个随机数，这些随机数分别遵循均值为10、20、60，
#标准差为2、5、10的正态分布。
#这样，我们就得到了一个包含5行4列的数据表，用于表示这5天的天气情况。
print(df)
```
## 2.数据聚合操作
```python
print(df.describe())#上节讲到的基本统计量

print(df.mean())#每列平均值

print(df.std())#每列标准差
```
## 3.时间处理
```python
#提取时间特征
df['年份']=df['日期'].dt.year
df['月份']=df['日期'].dt.month
df['星期']=df['日期'].dt.dayofweek
print(df)
```
## 4.数据分组
```python
#按月份分组并计算平均值
df['月份']=df['日期'].dt.month
grouped=df.groupby('月份').mean()#按月份分组的平均值
print(grouped）
```
## 5.数据透视表
```python
#创建更复杂的数据，加入风向
df_pivot = pd.DataFrame({
    '日期': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    '风速': np.random.normal(10, 2, 10),
    '温度': np.random.normal(20, 5, 10),
    '湿度': np.random.normal(60, 10, 10),
    '风向': np.random.choice(['东', '南', '西', '北'], 10)
})
df_pivot['月份']=df_pivot['日期'].dt.month

#创建透视表
#透视表（Pivot Table）是一种用于汇总、分析数据的工具，
#它可以将数据按照特定的行和列进行分组，
#并对每个分组进行聚合计算，如求和、平均、计数等。
pivot_table = pd.pivot_table(
    df_pivot,             # DataFrame对象，需要被透视的数据集
    values=['风速', '温度', '湿度'],  # 需要聚合的列名列表
    index='风向',          # 行分组依据的列名
    columns='月份',        # 列分组依据的列名
    aggfunc='mean'         # 聚合函数，这里使用的是求平均值
)

print(pivot_table)#打印风向和月份的透视表
```
## 6.数据清洗（重要！）
```python
#创建包含缺失值的数据
df_clean=df.copy()#创建一个原始数据框df的副本。
df_clean.loc[0,'风速']=np.nan#将第0行的'风速'列的值修改为缺失值（NaN），
df_clean.loc[2,'温度']=np.nan#逻辑同上
#这样做的目的是为了在不影响原始数据框df的情况下，
#对df_clean进行数据清洗或处理。
print(df_clean)

#处理缺失值
print(df_clean.fillna(df_clean.mean()))#使用平均值填充缺失值

#删除包含缺失值的行
print(df_clean.dropna())
```
## 7.数据合并
```python
#创建两个DataFrame
df1=pd.DataFrame({
    '日期':dates,
    '风速':np.random.normal(10,2,5)
})
df2=pd.DataFrame({
    '日期':dates,
    '温度':np.random.normal(20,5,5)
})

#合并数据
merged_df=pd.merge(df1,df2,on='日期')
#将两个数据框 df1 和 df2 合并成一个新的数据框 
#merged_df，合并的依据是它们共有的列 ‘日期’
print(merged_df)#打印合并后的数据
```
## 8. 数据导出和导入
```python
#导出到csv
df.to_csv('weather_data.csv',index=False)

#从scv导入
df_imported=pd.read_csv('weather_data.csv')
print(df_imported)#打印打入的数据
```
请运行上述代码，并尝试：
1. 修改一些参数（如日期范围、随机数据的分布等）
2. 添加新的列或计算
3. 尝试不同的数据清洗方法
4. 尝试不同的数据合并方式
动手自己敲一遍，学习的更扎实，看一遍是记不住什么的，自己敲一遍还能留到电脑里，说不定哪天就ctrl+c，ctrl+v上了呢？哈哈哈哈，前进吧！