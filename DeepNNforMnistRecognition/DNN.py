# -*- coding: utf-8 -*-

# 功能：构建深度神经网络并实现MNIST手写数字识别

import numpy as np
import os
import struct

def load_data(path, kind='train'):
    # 加载数据集，kind用来标识训练集/测试集
    images_file = os.path.join(path, '%s-images.idx3-ubyte' % kind)  # 图片文件名
    labels_file = os.path.join(path, '%s-labels.idx1-ubyte' % kind)  # 标签文件名

    # 以二进制方式读取图片文件
    with open(images_file, 'rb') as ifile:
        '''
        num : 样本数量
        rows/cols : 每个样本图片的像素点数
        '''
        magic, num, rows, cols = struct.unpack('>IIII', ifile.read(16))
        # 将图片的像素值展开为一个rows*cols的向量，图片数据存储为(num, rows*cols)的矩阵
        images = np.fromfile(ifile, dtype=np.uint8).reshape(num, rows*cols)

    # 以二进制方式读取标签文件
    with open(labels_file, 'rb') as lfile:
        magic, num = struct.unpack('>II', lfile.read(8))
        labels = np.fromfile(lfile, dtype=np.uint8)

    return images.T, labels

def label2vec(labels):
    '''
    # 将标签值转化为向量,例如标签 '5' 表示为 [0,0,0,0,0,1,0,0,0,0]
    '''
    y = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        y[labels[i], i] = 1
    y[np.where(y == 0)] = 0.001
    y[np.where(y == 1)] = 0.999
    return y

def vec2label(vec):
    return vec.argmax(axis=0)  # 每一列的最大值的索引就是对应的标签

def init_parameters(layer_dims):
    L = len(layer_dims)  # 获取网络层数+1
    parameters = {}
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1   # 每一层的权重
        parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))   # 每一层的阈值

    return parameters

def relu(z):
    return np.maximum(z, 0)

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    a[np.where(a == 1)] = 0.999
    a[np.where(a == 0)] = 0.001
    return a

def linear_activation_forward(A_prev, W, b, activation):
    Z = None
    A = None
    # 隐藏层使用relu激活函数
    if activation == 'relu':
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
    # 输出层使用sigmoid函数
    if activation == 'sigmoid':
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)
    cache = (A_prev, Z, W)
    return A, cache

def model_forward(X, paras):
    '''
    前向计算得到样本预测值
    :param X: 样本特征
    :param paras: 网络参数(W和b)
    :return:
        AL : 样本预测值
        caches : 计算中间过程的缓存值(Al 和 Zl)
    '''
    caches = []
    L = len(paras) // 2   # 网络层数
    A = X
    # 根据前一层激活值计算本层激活值，重复L-1次
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, paras['W'+str(l)], paras['b'+str(l)], 'relu')  # 前面用ReLU激活函数
        caches.append(cache)
    AL, cache = linear_activation_forward(A, paras['W'+str(L)], paras['b'+str(L)], 'sigmoid')   # 最后一层用sigmoid激活函数
    caches.append(cache)
    return AL, caches

def calculate_cost(Y_hat, Y):
    m = Y.shape[1]  # 样本数
    cost = -np.sum(np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat)))
    cost = np.squeeze(cost / m)
    return cost

def linear_activation_backward(dA, cache, activation):
    # cache 中包含（A_prev, Z, W）
    dA_prev = None
    dW = None
    db = None
    m = dA.shape[1]
    A_prev, Z, W = cache
    if activation == "sigmoid":
        dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))
        dW = np.dot(dZ, A_prev.T) / m
        db = np.mean(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    if activation == "relu":
        dZ = dA
        dW = np.dot(dZ, A_prev.T) / m
        db = np.mean(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = np.divide(1-Y, 1-AL) - np.divide(Y, AL)
    current_cache = caches[-1]  #(ZL, WL, AL-1)
    # 最后一层的梯度
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    # 反向计算前L-2层的梯度
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_paras(paras, grads, learning_rate):
    L = len(paras) // 2
    for l in range(1, L+1):
        paras["W" + str(l)] = paras["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        paras["b" + str(l)] = paras["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return paras

def model(X, Y, layer_dims, iteration, learning_rate):
    parameters = init_parameters(layer_dims)  # 参数初始化

    for t in range(iteration):
        AL, caches = model_forward(X, parameters)  # 前向计算预测值
        cost = calculate_cost(AL, Y)  # 计算损失函数值
        grads = model_backward(AL, Y, caches)  # 计算每一层的梯度值
        parameters = update_paras(parameters, grads, learning_rate)  # 更新网络参数

        # if t % 100 == 0:
        print("The %sth iteration, cost is %s" % (t, cost))

    return parameters

def predict(X, classifier):
    Y, caches = model_forward(X, classifier)  # 计算输出值
    labels = vec2label(Y)   # 将输出向量转化为标签
    return labels

def get_accuracy_rate(label1, label2):
    # 将两个标签向量相减，零元素的个数占比即为准确率
    diff = label1 - label2
    index = np.where(diff == 0)
    return index[0].size / len(label1)

if __name__ == '__main__':
    training_x, training_labels = load_data('mnist', 'train')  # 训练集样本和标签
    test_x, test_labels = load_data('mnist', 't10k')  # 测试集样本和标签
    # print(np.where(training_x != 0))
    # 将[0,255]区间内的像素值归一化到[0,1]
    training_x = training_x / 255
    test_x = test_x / 255
    # 将标签值转化为向量
    training_y = label2vec(training_labels)
    test_y = label2vec(test_labels)
    n_x = training_x.shape[0]  # 输入层神经元个数
    n_y = test_y.shape[0]      # 输出层神经元个数
    # print(training_x.shape)
    # print(training_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)

    # 构建并训练网络
    learning_rate = 0.05  # 学习率
    iteration = 1000     # 最大训练代数
    layer_dims = [n_x, 200, 100, 50, n_y]  # 构建深层神经网络，输入输出层神经元数量由样本决定

    # 训练分类器
    classifier = model(training_x, training_y, layer_dims, iteration, learning_rate)

    # 在训练集上测试准确率
    training_prediction = predict(training_x, classifier)
    accuracy_rate1 = get_accuracy_rate(training_labels, training_prediction)
    print("Accuracy on training set is %s." % accuracy_rate1)

    # 在测试集上测试准确率
    test_prediction = predict(test_x, classifier)
    accuracy_rate2 = get_accuracy_rate(test_labels, test_prediction)
    print("Accuracy on test set is %s." % accuracy_rate2)