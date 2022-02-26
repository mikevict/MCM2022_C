

import pandas as pd

max_best_days_g = []
max_best_days_b = []
min_best_days_g = []
min_best_days_b = []


# 获取模型当天的交易额
def prediction(count, coef: list, intercept, a):
    res = intercept
    # print('第几次交易：', count)
    for i, c in enumerate(coef):
        res += c * pow(count, 1 + i)
    return res * (1 - a)

#几个较大点，几个较小点
def get_point(file_path, coef: list, intercept, a, num=3):
    d = dict()
    data = pd.read_csv(file_path)
    y_data = data.iloc[1:, 2]
    for i in range(len(y_data)):
        d[i] = y_data[i + 1]
    #从小到大排序根据时间
    d = sorted(d.items(), key=lambda x: x[1])
    # print('前几个较大的交易点', d[-num:])
    # print('前几个较小的交易点', d[: num])
    global max_best_days_g, min_best_days_g, max_best_days_b, min_best_days_b
    if 'GOLD' in file_path:
        max_best_days_g = d[-num:]
        min_best_days_g = d[:num]
    else:
        max_best_days_b = d[-num:]
        min_best_days_b = d[:num]

    # 把点传入模型进行计算,max_w卖出， min_w买入
    #？
    max_w, min_w = 0, 0
    for e in d[-num:]:
        max_w += prediction(e[0], coef, intercept, a)
    for e in d[:num]:
        min_w += prediction(e[0], coef, intercept, a)

    return max_w - min_w


# 找到最佳的交易时间
def find_trading_days(data_path_g, data_path_b):
    g_data = pd.read_csv(data_path_g).iloc[1:, 1]
    b_data = pd.read_csv(data_path_b).iloc[1:, 1]
    res_date_sale_g = []
    res_date_sale_b = []
    res_date_buy_g = []
    res_date_buy_b = []

    for i in max_best_days_g:
        res_date_sale_g.append(g_data[i[0]])
    for i in max_best_days_b:
        res_date_sale_b.append(b_data[i[0]])
    for i in min_best_days_g:
        res_date_buy_g.append(g_data[i[0]])
    for i in min_best_days_g:
        res_date_buy_b.append(b_data[i[0]])
    print('最佳黄金卖出时间', res_date_sale_g)
    print('最佳黄金买出时间', res_date_buy_g)
    print('最佳比特币卖出时间', res_date_sale_b)
    print('最佳比特币买出时间', res_date_buy_b)


def train(file_path, coef: list, intercept, a, r=11):
    dw = dict()
    for i in range(1, r):
        w = get_point(file_path, coef, intercept, a, i)
        dw[i] = w
    print(dw)
    dw = sorted(dw.items(), key=lambda x: x[1])
    best_num = max(dw)[0]
    best_w = max(dw)[1]
    print(best_num, best_w)
    return best_num, best_w


if __name__ == '__main__':
    bit_coef = [-5.52957703e-04, -4.65922448e-02, 3.80701831e-04,
                -6.85859284e-07, 4.58595999e-10, -1.02636974e-13]
    gold_coef = [5.30147327e-05, 8.40271443e-03, -4.23485209e-05,
                 7.68679724e-08, -5.74630762e-11, 1.51678800e-14]
    bit_intercept = 1433.8246533975998
    gold_intercept = 1190.3827674894403

    # 下面是找3个较大和3个较小的点的例子
    print('3个较大和3个较小的点的例子')
    gold_w = get_point('./data/LBMA-GOLD_new.csv', gold_coef, gold_intercept, 0.01, 3)
    print('黄金总收益:', gold_w)

    bitcoin_w = get_point('./data/BCHAIN-MKPRU_new.csv', bit_coef, bit_intercept, 0.02, 3)
    print('比特币总收益:', bitcoin_w)

    W = gold_w + bitcoin_w - 1000
    print('总收益:', W, 'dollars\n')

    # 下面是自动找多个较大和较小的点，控制范围是r，也就是点的个数，可以自己调整
    print('多个较大较小点的例子\n')
    best_num1, gold_w = train('./data/LBMA-GOLD_new.csv', gold_coef, gold_intercept, 0.01, r=11)
    best_num2, bitcoin_w = train('./data/BCHAIN-MKPRU_new.csv', bit_coef, bit_intercept, 0.02, r=11)
    print('黄金总收益:', gold_w)
    print('比特币总收益:', bitcoin_w)
    print('黄金最佳的选点方案的交易次数：', best_num1)
    print('比特币最佳的选点方案的交易次数：', best_num2)

    W = gold_w + bitcoin_w - 1000
    print('总收益:', W, 'dollars')

    find_trading_days('./data/LBMA-GOLD_new.csv', './data/BCHAIN-MKPRU_new.csv')
