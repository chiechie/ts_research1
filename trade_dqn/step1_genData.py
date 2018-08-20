# -*- coding: utf-8 -*-
import os

from numpy import genfromtxt
import numpy as np

from common.path_helper import savePklto, join, list_md5_string_value, loadPklfrom
from trade_dqn.supervised_helper import generate_actions_from_price_data
from settings import data_dict_path, data_path, supervised_y_data_path, raw_data_file, moving_average_number



def prepare_data(path, data_path, data_dict_path):
    stock_data = genfromtxt(path, delimiter=',', dtype=None, names=True)
    average_dataset = []
    total_data = []
    temp_episode = []
    data_dict = {}
    index = 0
    lines = []
    print("total ", stock_data)
    for data in stock_data:
        # temp = [open, high, low, close, count]
        # dict_vector = temp + [average], but average is incorrect.
        # len(average_dataset) == 1002
        temp = [data[2], data[3], data[4], data[5], data[8]]
        average_dataset.append(temp)
        if index % 1000 == 13:
            print(index, len(average_dataset))
        # print(index)
        # print(len(average_dataset))
        if index > moving_average_number:
            # average_dataset始终保持长度 == moving_average_number
            # mean是average_dataset的5个特征的均值
            mean = np.mean(average_dataset, axis=0)
            # mean_array是将average_dataset按照mean给scale之后的结果
            mean_array = average_dataset / mean
            # last_minute_data是原始数据归一化之后最近1分钟的均值
            last_minute_data = mean_array[-1]
            # last_one_hour_average是原始数据归一化之后最近一个小时的均值
            last_one_hour_average = np.mean(mean_array[-60:], axis=0)
            # last_one_hour_average是原始数据归一化之后最近一天的均值
            last_one_day_average = np.mean(mean_array[-300:], axis=0)
            # last_one_hour_average是原始数据归一化之后最近三天的均值
            last_3_day_average = np.mean(mean_array[-900:], axis=0)  # this might change

            # average_dataset 扔掉旧数据
            average_dataset = average_dataset[1:]

            # 将归一化后的4个window_size * 5个feature的MA（一共20个）先装入vector
            # 再将vector装入data
            # 使用vector的向量值 作为key
            # 该时刻的原始价格（4个值） +  平均值（1个值） 为 value
            # 将 (key , value)对装入data_dict
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
    savePklto(total_data, data_path)
    savePklto(data_dict, data_dict_path)
    return None


def make_supervised_data(data_path, data_dict_path, supervised_y_data_path):
    data = loadPklfrom(data_path)
    data_dict = loadPklfrom(data_dict_path)
    supervised_data = []
    for episode in data:
        prices = [data_dict.get(list_md5_string_value(episode_i), [0])[-1] for episode_i in episode]
        assert len(prices) == 10
        golden_actions, final_profit = generate_actions_from_price_data(prices)
        supervised_data.append(golden_actions)
    savePklto(supervised_data, supervised_y_data_path)
    return None


def check_dimension(data_path, supervised_y_data_path):
    data = loadPklfrom(data_path)
    supervised_y_data = loadPklfrom(supervised_y_data_path)
    print("len(data)", len(data))
    print("len(supervised_y_data)", len(supervised_y_data))


if __name__ == "__main__":
    # prepare_data(raw_data_file, data_path, data_dict_path)
    # make_supervised_data(data_path, data_dict_path, supervised_y_data_path)
    check_dimension(data_path, supervised_y_data_path)

