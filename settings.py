# -*- coding: utf-8 -*-
# Standard library
# Third Party Library
# My Library
from collections import OrderedDict
import os
from os.path import join

default_config = {}
addition_config = {}
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# raw_data_file = os.path.join(parent_dir, 'tensor-reinforcement/NIFTY50.csv')
raw_data_file = os.path.join(parent_dir, './test_data/ib/csv_data/AJP.csv')
supervised_y_data_path = join(parent_dir, "./test_data/rf_test_data/supervised_data.pkl")
data_path = join(parent_dir, "./test_data/rf_test_data/data.pkl")
data_dict_path = join(parent_dir, "./test_data/rf_test_data/data_dict.pkl")
moving_average_number = 1000  # number of time interval for calculating moving average


pre = '.'
def get_user_data_dir():
    config_json = Config_json()
    if os.getcwd().startswith('/Users/stellazhao'):
        LOCAL_UDF_DATA_DIR = config_json.get_config('LOCAL_UDF_DATA_DIR')['shihuanzhao']
    elif os.getcwd().startswith('/data/mapleleaf'):
        LOCAL_UDF_DATA_DIR = config_json.get_config('REMOTE_DATA_DIR')
    elif os.getcwd().startswith('/Users/brent/'):
        LOCAL_UDF_DATA_DIR = config_json.get_config('LOCAL_UDF_DATA_DIR')['gaojian']
    else:
        print('get_user_data_dir error ')
        LOCAL_UDF_DATA_DIR = ''
    print('your local data dir prefix is: %s' % (LOCAL_UDF_DATA_DIR))
    return LOCAL_UDF_DATA_DIR


override_config = {
    'ENV_TYPE': 'TEST',
    ##PATH CONFIG
    'LOCAL_UDF_DATA_DIR': {
        'shihuanzhao':
            '/Users/stellazhao/tencent_workplace/labgit/dataming/ts_research1/test_data',
	'gaojian':'/Users/brent/Downloads/ts_research1/test_data'	
        },
    "REMOTE_DATA_DIR": "/data/mapleleaf/shihuan/",
    ### svm test ###
    "original_data": pre + "/order_book_DB/",
    "STEP1_DATA_SUBDIR": pre + "/labeled_data/",
    "STEP1_PIC_SUBDIR": pre + "/step1_labeled_data_pic/",

    "STEP2_DATA_SUBDIR": pre + "/step2_model_data",
    "STEP2_PIC_SUBDIR": pre + "/step2_transform_pic",
    ### reinforcement test data ###
    "rf_test_data": pre + "/rf_test_data",

    ##ENV CONFIG
    #"ENV": "REMOTE",
    "ENV": "LOCAL",

    ## ALGORITHM PARAMETERS
    "LABEL_CODE_DICT": {
            "upward": 1,
              "downward": -1,
              "stable": 0
              },
    "PREDICT_LENGTH_POINTS": 15,
    "DELTA_T1_POINTS": 10,
    "DELTA_T2_POINTS": 900,
    "TRAIN_POINTS": 1500,
    "TICK_SEC": 3,
    "TRAIN_UPDATE_SEC": 60 * 60,


}


def get_addition_cfg(biz_id, case_type):
    try:
        res = addition_config[biz_id][case_type]
    except:
        # print '[CONFIG] got empty addition cfg'
        res = {}
    return res


class Config_json():
    def __init__(self, biz_id='DEFAULT', case_type='DEFAULT'):
        self.biz_id = biz_id
        self.case_type = case_type
        self.default_config = default_config
        self.biz_case_config = get_addition_cfg(self.biz_id, self.case_type)
        self.override_config = override_config
        self.final_config = dict()
        self.final_config.update(self.default_config)
        self.final_config.update(self.biz_case_config)
        self.final_config.update(self.override_config)
        self.final_config.update({'BIZ_ID': self.biz_id, 'CASE_TYPE': self.case_type})

    def get_config(self, value):
        try:
            res = self.final_config[value]
        except:
            res = None
        return res




