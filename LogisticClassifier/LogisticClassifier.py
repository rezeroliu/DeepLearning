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

def forward(X, Y, w, b):
    m = X.shape[1]   # 样本数量
    Y_hat = sigmoid(np.dot(w.T, X))  # 根据当前权重w和阈值b计算预测值Y_hat
    cost = -(np.dot(Y, np.log(Y_hat.T)) + np.dot(1-Y, np.log(1-Y_hat).T)) / m    # 交叉熵作为损失函数，计算损失值
    # 计算w和b的梯度
    dw = np.dot(X, (Y_hat-Y).T) / m
    db = np.mean(Y_hat - Y)

    return cost, dw, db

def model(X, Y, iterations, learning_rate):
    # 初始化w和b
    w = np.random.rand(X.shape[0],1)    # 随机初始化
    # w = np.zeros((X.shape[0],1))   # 初始化为0
    '''
    实验发现随机初始化在训练集上的准确率总是比初始化为0低，但是多次实验表明随机初始化生成的模型在测试机上的正确率总是较高
    '''
    b = 0
    cost_list = []  # 存储每一次迭代的损失值

    # 优化w和b
    for i in range(iterations):
        # 前向传播，计算Y_hat、cost和grads
        cost, dw, db = forward(X, Y, w, b)
        # 更新w和b
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # 将cost添加到损失值列表
        cost_list.append(cost)

        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    classifier = {
        "weight" : w,
        "bias" : b
    }
    return classifier

def predict(params, X):
    w = params["weight"]
    b = params["bias"]
    Y = sigmoid(np.dot(w.T, X))
    # 将Y转化为0，1二元值
    Y[np.where(Y < 0.5)] = 0
    Y[np.where(Y > 0.5)] = 1

    return Y

if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()  # 加载数据
    print(train_x_orig.shape) # 图片数据都是width=height的图片
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
    learning_rate = 0.005  # 学习率
    classifier = model(train_x, train_y, iterations, learning_rate) # 训练参数得到分类器参数
    # 利用训练出的分类器进行分类
    predictionY_train = predict(classifier, train_x)
    predictionY_test =predict(classifier, test_x)

    # 计算训练集和测试集的准确率
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



