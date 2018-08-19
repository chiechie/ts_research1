# coding=utf-8
# 根据股价序列，计算出最优的执行策略
import numpy as np

action_list = [1, 2]


def find_profit_from_given_action(price_list):
    long_position = [1 if l < price_list[-1] else -1 for l in price_list[:-1]]
    long_position = np.array(long_position)
    final_profit = -np.sum(long_position * np.array(price_list)[:-1]) + np.sum(long_position) * price_list[-1]
    return long_position, final_profit


def test():
    price_list = [1, 2, 600, 4, 5, 6,7,8,9,5]
    print(find_profit_from_given_action(price_list))


if __name__ == "__main__":
    test()