# -*- coding: utf-8 -*-

# 功能：实现一个单隐层的神经网络分类器

import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py  # 读取 .h5格式的数据集
import scipy
from scipy import ndimage

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')   #训练集数据
    train_x_orig = np.array(train_dataset["train_set_x"][:])    # 训练集特征
    train_y_orig = np.array(train_dataset["train_set_y"][:])   # 训练集标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')  # 测试集数据
    test_x_orig = np.array(test_dataset["test_set_x"][:])  # 测试集特征
    test_y_orig = np.array(test_dataset["test_set_y"][:])  # 测试集标签

    classes = np.array(test_dataset["list_classes"][:])  # 种类列表（1:cat or 0:non-cat）

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X, paras):
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)            # 隐层输出
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)            # 网络输出值

    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2}
    return cache

def compute_cost(X, Y, Y_hat):
    m = Y.shape[1]  # 样本数量
    logprobs = Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)  #交叉熵
    cost = -np.sum(logprobs) / m
    return cost

def backpropagation(paras, cache, X, Y):
    W2 = paras["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    m = X.shape[1]  # 样本数
    # 反向传播误差计算梯度
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, Y.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2}
    return grads

def update_parameters(paras, grads, learning_rate):
    # 当前参数
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    # 梯度值
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # 梯度下降法更新
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}
    return  parameters


def init_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01  # 输入层到隐层的权重
    # W1 = np.zeros((n_h, n_x))
    b1 = np.zeros((n_h, 1))                # 隐层的阈值
    W2 = np.random.randn(n_y, n_h) * 0.01  # 隐层到输出层的权重
    # W2 = np.zeros((n_y, n_h))
    b2 = np.zeros((n_y, 1))                # 输出层的阈值

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2}

    return parameters

def model(X, Y, n_h, iterations, learning_rate):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    # 初始化参数w和b
    parameters = init_parameters(n_x, n_h, n_y)
    cost_list = []  # 存储每一次迭代的损失值


    # 优化w和b
    for i in range(iterations):
        # 前向传播，计算Y_hat、cost和grads
        cache = forward_propagation(X, parameters)  # 前向迭代计算各层输出
        cost = compute_cost(X, Y, cache["A2"])   # 计算损失值
        cost_list.append(cost)
        grads = backpropagation(parameters, cache, X, Y)  # 计算梯度
        parameters = update_parameters(parameters, grads, learning_rate)  # 更新网络参数

        if i % 100 == 0:
            print("The %sth iterations, cost is %s" % (str(i), str(cost)))

    return parameters

def predict(paras, X):
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    A1 = np.tanh(np.dot(W1, X) + b1)  # 隐层输出值
    Y = sigmoid(np.dot(W2, A1) + b2)  # 预测值
    return Y

if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()  # 加载数据
    num_px = train_x_orig.shape[1]
    # 数据处理，将每个样本的特征表示为一个列向量，处理后的二维矩阵形状为(特征数*样本数)
    train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # train_x_orig.shape[0]表示训练样本数
    test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # 标准化到[-1,1], 255表示最大像素值
    train_x = train_x / 255
    test_x = test_x / 255
    # 标签值为 (1*样本数) 的行向量
    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

    # Logistic分类器
    iterations = 2000  # 最大迭代次数
    learning_rate = 0.01  # 学习率
    n_h = 10   # 隐藏层节点数

    # 训练出的分类器
    classifier = model(train_x, train_y, n_h, iterations, learning_rate)

    # 在训练集和测试集上计算训练出分类器的分类准确率
    predictionY_train = predict(classifier, train_x)
    predictionY_test = predict(classifier, test_x)
    print("traing_accuracy is: {} %".format(100 - np.mean(np.abs(predictionY_train - train_y)) * 100))
    print("test_accuracy is: {} %".format(100 - np.mean(np.abs(predictionY_test - test_y)) * 100))

    # 用自己的图片进行测试
    fname = 'images/my_image.jpg'
    image = np.array(ndimage.imread(fname, flatten=False))  # 读取图片
    # 将图片转化为(1, num_px * num_px * 3)的列向量以匹配神经元的输入
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(classifier, my_image)

    plt.imshow(image)
    pylab.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")