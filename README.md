# xgboost Emotional translation

## 简介

用xgboost进化树制作的可以根据文本文字判断文字中附带的人类情感。附带模型文件，也可以根据函数自己进行
训练，可以自己进行调参。作者训练出来的模型准确率并不高。

## 前提

安装了python解释器

安装了git

## 安装

### 克隆仓库到本地

在本地电脑创建一个文件夹用来存储该项目文件，然后在该文件加中打开终端，运行下面的代码。
Windows:

```bash
git clone https://github.com/dh31223/Emotional-translation.git
```

### 更换软件下载源

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 数据下载

```bash
打开
https://tianchi.aliyun.com/dataset/198371?accounttraceid=b57695d9ba324bac918ca7a5c0e9cf8feflk
下载数据
```

如果想直接运行，请把数据以csv形式存储在data目录中。

### 创建Python虚拟环境

```bash
python -m venv .venv
```

### 下载相应的python包

```bash
pip install -r resquirement.txt
```

### 运行predict可以查看测试结果

```bash
python predict.py
```

## 功能

### predict.py中的函数

load_model() 导入模型和TF-IDF

tokenize() 对输入的完整句子用jieba进行分割

predict() 用来预测句子所带的情绪

### train.py中的函数

load_data() 用来导入训练数据

fit_tfidf_vectorizer() 用来训练TF-IDF转换器

train_model() 用来训练模型

GridSearchcv() 通过交叉验证网格搜索来寻找参数，config.py中已经存储了作者寻找的参数。


## 制作中遇到的困难和解决方案

问题1：用交叉验证网格搜索寻找最佳参数时，由于数据量大且参数网格复杂，需要耗费巨大的算力。

解决1：用分步交叉验证网格搜索，每次只进行3个参数的网格搜索，然后分三次，最终确定最好的参数组。

问题2：第一版的py程序是把26万条数据用train_test_split进行分割的，text_size = 0.15，但是我在分割之前先把这些数据用于TF-IDF的训练，导致数据泄露，出现了模型在测试集%100的正确率。

解决2：先分割，将训练集用来给TF-IDF训练，测试集没有被用来训练，避免了数据泄露。

问题3：把TF-IDF的max_features调的过低，导致模型欠拟合，从而浪费了大部分算力。

解决3：把max_features从3000提升到20000，并且相应的提升了xgboost的n_estimators和max_depth和colsample_bytree。

