import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # 生成多项式用的
import matplotlib.pyplot as plt


def handle_bitcoin(data_path):
    data = pd.read_csv(data_path)
    # 用缺失值上面的值替换缺失值
    data = data.fillna(axis=0, method='ffill')
    data.to_csv('./data/BCHAIN-MKPRU_new.csv')


def show_bitcoin(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 1]
    y_data = data.iloc[1:, 2]
    plt.scatter(x_data, y_data, s=1)
    plt.show()


def build_model(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 0]
    #  print(type(x_data))
    y_data = data.iloc[1:, 2]
    # 转换为二维数据
    # degree = n，相当于n次方拟合
    poly = PolynomialFeatures(degree=6)
    # 特征处理
    #转换x_data的类型为数组
    x_data = np.array(x_data).reshape((len(x_data), 1))
    #   print(type(x_data))
    x_poly = poly.fit_transform(x_data)
    model = LinearRegression()

    model.fit(x_poly, y_data)
    print('系数：', model.coef_)
    print('截距：', model.intercept_)

    # 画图
    plt.scatter(x_data, y_data, s=1)
    plt.plot(x_data, model.predict(x_poly), 'r')  # predict 传的是x_poly,是处理后的数据
    plt.title('Polynomial Regression BCHAIN-MKPRU Model')
    plt.xlabel('date/days')
    plt.ylabel('dollars$ /bitcoin')
    plt.show()

    return len(y_data), model.coef_[1:], model.intercept_


# 计算最初的 1000 美元投资价值
def prediction(init_money, count, coef: list, intercept, a=0.02):
    """
    :param init_money: 最初的投资
    :param count: 第几次交易
    :param coef: 模型系数
    :param intercept: 模型截距
    :param a: αbitcoin = 2%
    :return: init_money在第days天的投资价值
    """
    res = intercept
    print('第几次交易：', count)
    for i, c in enumerate(coef):
        res += c * pow(count, 1 + i)
    print('比特币每日价格(dollars per bitcoin): ', res, '美元$')
    return init_money / res * (1 - a)


if __name__ == '__main__':
    # handle_bitcoin('./data/BCHAIN-MKPRU.csv')
    # show_bitcoin('./data/BCHAIN-MKPRU_new.csv')
    count, coef, intercept = build_model('./data/BCHAIN-MKPRU_new.csv')
    print('2021 年 9 月 10 日,最初的 1000 美元投资价值:', prediction(1000, count, coef, intercept), '比特币')
