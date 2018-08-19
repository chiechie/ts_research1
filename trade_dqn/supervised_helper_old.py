# -*- coding: utf-8 -*-
# 根据股价序列，计算出最优的执行策略
# 原始代码的逻辑是
# 1.先使用递归函数生成所有策略
# 2.在所有策略中找到最优的一组策略
# 个人觉得，这种方法复杂且没有必要：
# 都已经知道每个时刻价格了，难道最优的执行策略不是能直接写出来吗：
# 比最终价格低/高的时刻short/long
# 好傻逼啊。。。

total_iteration_list = []
profit = 0
action_list = [1, 2]
episode = 9
global_array = []


def get_iteration_actions_recursive(action_list, temp_array, episode, global_array):
    if episode == 0:
        global_array.append(temp_array)
        return global_array
    for action in action_list:
        new_temp_array = temp_array + [action, ]
        get_iteration_actions_recursive(action_list, new_temp_array, episode - 1, global_array)
    pass


get_iteration_actions_recursive(action_list, [], episode, global_array)


def generate_actions_from_price_data(prices):
    """
    :param prices: 股价序列
    :return:
    golden_actions:最优的执行策略
    """
    old_profit = 0
    golden_actions = []
    for action_list in global_array:
        # the below method can also be replaces with algorithms
        # which generates action non-iteravely but
        #i am too lazy to do that and got lot of computation power
        profit, result_list = find_profit_from_given_action(prices, action_list)
        if profit >= old_profit:
            old_profit = profit
            golden_actions = result_list
    return golden_actions, old_profit


def find_profit_from_given_action(prices, actions):
    portfilio = 0
    portfilio_value = 0
    result_list = []
    for index, action in enumerate(actions):
        price = prices[index]
        if action == 1: #buy
            portfilio += 1
            portfilio_value -= price
        elif action == 2: #sell
            portfilio -= 1
            portfilio_value += price
        result_list.append([action, portfilio])
    profit = portfilio_value + (portfilio) * prices[-1]
    return profit, result_list


if __name__ == "__main__":
    price_list = [1, 2, 600, 4,5,6,7,8,9,5]
    a = generate_actions_from_price_data(price_list)
    print(a)