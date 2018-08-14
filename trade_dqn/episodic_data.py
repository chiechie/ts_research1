# -*- coding: utf-8 -*-
import gzip
import os

from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
import numpy as np
import cPickle as pickle
import hashlib
import json
from supervised_helper import generate_actions_from_price_data

episode = 10  # length of one episode
data_array = []
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_data_file = os.path.join(parent_dir, 'tensor-reinforcement/NIFTY50.csv')
moving_average_number = 1000  # number of time interval for calculating moving average


def prepare_data(path):
    stock_data = genfromtxt(path, delimiter=',', dtype=None, names=True)
    average_dataset = []
    total_data = []
    temp_episode = []
    data_dict = {}
    index = 0
    lines = []
    print "total ", stock_data
    for data in stock_data:
        # temp = [open, high, low, close, count]
        # dict_vector = temp + [average], but average is incorrect.
        temp = [data[2], data[3], data[4], data[5], data[8]]
        average_dataset.append(temp)
        if index % 1000 == 13:
            print index, len(average_dataset)
        # print(index)
        # print(len(average_dataset))
        if index > moving_average_number:
            # average_dataset始终保持长度 == moving_average_number
            #mean是average_dataset的5个特征的均值
            mean = find_average(average_dataset)
            # mean_array是将average_dataset按照mean给scale之后的结果
            mean_array = average_dataset / mean
            # last_minute_data是原始数据归一化之后最近1分钟的均值
            last_minute_data = mean_array[-1]
            # last_one_hour_average是原始数据归一化之后最近一个小时的均值
            last_one_hour_average = find_average(mean_array[-60:])
            # last_one_hour_average是原始数据归一化之后最近一天的均值
            last_one_day_average = find_average(mean_array[-300:])
            # last_one_hour_average是原始数据归一化之后最近三天的均值
            last_3_day_average = find_average(mean_array[-900:])  # this might change

            # average_dataset 扔掉旧数据
            average_dataset = average_dataset[1:]

            #将归一化后的4个window_size * 5个feature的MA（一共20个）装入data
            #同时将新进入的一条记录的5个特征(未做归一化的) 和 4个价格的均值存入data_dict
            vector = []
            vector.extend(last_minute_data)
            vector.extend(last_one_hour_average)
            vector.extend(last_one_day_average)
            vector.extend(last_3_day_average)
            # What's the average_price here means? I don't think it is correct.
            average_price = sum(temp[0:-2]) / float(len(temp[0:-2]))
            dict_vector = temp + [average_price]
            md5 = list_md5_string_value(vector)
            lines.append("key:{0}\nmd5:{1}\nvalue:{2}".format(vector, md5, dict_vector))
            data_dict[md5] = dict_vector
            total_data.append(vector)
        index += 1
    with open("data.pkl", "wb") as myFile:
        # six.moves.cPickle.dump(total_data, myFile, -1)
        pickle.dump(total_data, myFile, -1)
    with open("data_dict.pkl", "wb") as myFile:
        # six.moves.cPickle.dump(data_dict, myFile, -1)
        pickle.dump(data_dict, myFile, -1)
    with open("debug.txt", "w") as txtFile:
        for line in lines:
            txtFile.write(line + "\n")


def find_average(data):
    return np.mean(data, axis=0)


def load_data(file, episode):
    data = load_file_data(file)
    return map(list, zip(*[iter(data)] * episode))


def load_file_data(file):
    with open(file, 'rb') as myFile:
        # data = six.moves.cPickle.load(myFile)
        data = pickle.load(myFile)
    return data


def list_md5_string_value(list):
    string = json.dumps(list)
    return hashlib.md5(string).hexdigest()


def episode_supervised_data(data, data_dict):
    prices = []
    for iteration in data:
        prices.append(data_average_price(data_dict, iteration))
    actions = generate_actions_from_price_data(prices)
    return actions


def data_average_price(data_dict, data):
    #data = data_dict[list_md5_string_value(data)]
    key = list_md5_string_value(data)
    value = data_dict.get(key)
    if value is None:
        return 0
    return value[-1]


def make_supervised_data(data, data_dict):
    supervised_data = []
    if os.path.exists('supervised_data.pkl'):
        with open('supervised_data.pkl', 'rb') as myFile:
            supervised_data = pickle.load(myFile)
        return supervised_data
    else:
        for episode in data:
            supervised_data.append(episode_supervised_data(episode, data_dict))
        with open("supervised_data.pkl", "wb") as myFile:
            pickle.dump(supervised_data, myFile, -1)
        return supervised_data
