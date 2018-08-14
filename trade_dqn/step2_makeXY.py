# -*- coding: utf-8 -*-
# Standard library
from os.path import join

# Third Party Library
import pandas as pd
from sklearn.cross_validation import train_test_split
try:
    import cPickle as pickle
except:
    import _pickle as pickle

# My Library
from common.path_helper import load_data, loadPklfrom

action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
data_dict = {}
# calling data_dict is data_dict[episodic_data.list_md5_string_value(list)]

def get_intial_data(data_path, data_dict_path, supervised_y_data_path):
    data = load_data(data_path, episode=10)
    # np.array(data).shape == 62773, 10, 20
    global data_dict
    supervised_y_data = loadPklfrom(supervised_y_data_path)
    x_train, x_test, y_train, y_test = train_test_split(data, supervised_y_data, test_size=0.10, random_state=123)
    data_dictionary = {}
    #here one is portfolio value
    data_dictionary["input"] = len(x_train[0][0]) + 1

    # short, buy and hold
    data_dictionary["action"] = 3
    data_dictionary["hidden_layer_1_size"] = 40

    # will be using later
    data_dictionary["hidden_layer_2_size"] = 20
    data_dictionary["x_train"] = x_train
    data_dictionary["x_test"] = x_test
    data_dictionary["y_test"] = y_test
    data_dictionary["y_train"] = y_train
    return data_dictionary


def new_stage_data(action, portfolio, old_state, new_state, portfolio_value, done, episode_data):
    # old_portfolio_value = portfolio_value
    #low_price = new_state[2]
    #changing code to use average price rather than normalized price
    global data_dict
    price = episodic_data.data_average_price(data_dict, episode_data)
    next_price = episodic_data.data_average_price(data_dict, new_state)
    #price = data_dict[episodic_data.list_md5_string_value(episode_data)][-1]
    #next_price = data_dict[episodic_data.list_md5_string_value(new_state)][-1]
    #buying
    if action == 1:
        #old_price = old_state[1]
        #Todo: Add transaction cost here also 
        portfolio_value -= price
        portfolio += 1
    #selling
    elif action == 2:
        #old_price = old_state[2]
         #Todo: Add transaction cost here also 
        portfolio_value += price
        portfolio -= 1

    elif action == 0:
        portfolio = portfolio
    #reward = 0
    #if new_state:
    new_state = new_state + [portfolio]
    #if portfolio >= 0:
        #low_price = new_state[2]
    #else:
        #low_price = new_state[1]
    #reward system might need to change and require some good thinking
    #if done:
    reward = (portfolio_value + portfolio * next_price)
    if reward > 0:
        reward = 2*reward #increasing reward
    #pdb.set_trace();
    return new_state, reward, done, portfolio, portfolio_value


def show_trader_path(actions, episode_data, portfolio_list, portfolio_value_list, reward_list):
    i = 0
    global data_dict
    #print("Action, Average Price, Portfolio, Portfolio Value, Reward")
    for index, action in enumerate(actions):
        episode = episode_data[index]
        action_name = action_map[actions[index]]
        price = episodic_data.data_average_price(data_dict, episode)
        portfolio = portfolio_list[index]
        portfolio_value = portfolio_value_list[index]
        i += 1
        reward = reward_list[index]
        #print(action_name, price, portfolio, portfolio_value, reward)
    #print("last price:")
    episode = episode_data[i]
    last_price = episodic_data.data_average_price(data_dict, episode)
    #print(last_price)
    reward = (portfolio_value_list[-1] + portfolio_list[-1]*last_price)
    return reward 