# coding=utf-8
# 根据股价序列，计算出最优的执行策略
import numpy as np

action_map = {1: 1,
			 -1: 2}

def generate_actions_from_price_data(price_list):
	long_position = [1 if l < price_list[-1] else -1 for l in price_list[:-1]]
	long_position = np.array(long_position)
	final_profit = -np.sum(long_position * np.array(price_list)[:-1]) + np.sum(long_position) * price_list[-1]
	position = np.cumsum(long_position)
	action_list = [action_map[l] for l in long_position]
	golden_actions = zip(action_list, position)
	golden_actions = [list(action) for action in golden_actions]
	return golden_actions, final_profit


def test():
    price_list = [1, 2, 600, 4, 5, 6,7,8,9,5]
    print(generate_actions_from_price_data(price_list))


if __name__ == "__main__":
    test()