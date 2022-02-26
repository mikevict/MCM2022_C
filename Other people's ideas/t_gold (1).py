import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # 生成多项式用的
import matplotlib.pyplot as plt


def handle_gold(data_path):
    data = pd.read_csv(data_path)
    # 处理空缺值，设为平均
    # data['USD (PM)'] = data['USD (PM)'].fillna(data['USD (PM)'].median())
    # 用缺失值上面的值替换缺失值
    data = data.fillna(axis=0, method='ffill')
    data.to_csv('./data/LBMA-GOLD_new.csv')


def show_gold(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 1]
    y_data = data.iloc[1:, 2]
    plt.scatter(x_data, y_data, s=1)
    plt.show()


def build_model(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 0]
    y_data = data.iloc[1:, 2]
    # 转换为二维数据
    # degree = n，相当于n次方拟合
    poly = PolynomialFeatures(degree=6)
    # 特征处理
    x_data = np.array(x_data).reshape((len(x_data), 1))
    x_poly = poly.fit_transform(x_data)
    model = LinearRegression()

    model.fit(x_poly, y_data)
    print('系数：', model.coef_)
    print('截距：', model.intercept_)

    # 画图
    plt.scatter(x_data, y_data, s=1)
    plt.plot(x_data, model.predict(x_poly), 'r')  # predict 传的是x_poly,是处理后的数据
    plt.title('Polynomial Regression LBMA-GOLD Model')
    plt.xlabel('date/days')
    plt.ylabel('dollars$ /troy ounce')
    plt.show()

    return len(y_data), model.coef_[1:], model.intercept_


# 计算最初的 1000 美元投资价值
def prediction(init_money, count, coef: list, intercept, a=0.01):
    #print('##')
    """
    :param init_money: 最初的投资
    :param days: 第几次交易
    :param coef: 模型系数
    :param intercept: 模型截距
    :param a: αgold = 1%
    :return: init_money在第days天的投资价值
    """
    res = intercept
    print('第几次交易：', count)
    for i, c in enumerate(coef):
        res += c * pow(count, i + 1)
    print('每金衡盎司美元(dollars per troy ounce): ', res, '美元$')
    return init_money / res * (1 - a)


if __name__ == '__main__':
    # handle_gold('./data/LBMA-GOLD.csv')
    # show_gold('./data/LBMA-GOLD_new.csv')
    count, coef, intercept = build_model('./data/LBMA-GOLD_new.csv')
    print('2021 年 9 月 10 日,最初的 1000 美元投资价值:', prediction(1000, count, coef, intercept), '盎司')
