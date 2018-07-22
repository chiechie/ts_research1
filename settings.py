# -*- coding: utf-8 -*-
# Standard library
# Third Party Library
# My Library
# -*- coding: utf-8 -*-
from collections import OrderedDict
import os

default_config = {}

addition_config = {}

pre = "test_ts"
pre = '.'
def get_user_data_dir():
    config_json = Config_json()
    if os.getcwd().startswith('/Users/stellazhao'):
        LOCAL_UDF_DATA_DIR = config_json.get_config('LOCAL_UDF_DATA_DIR')['shihuanzhao']
    elif os.getcwd().startswith('/data/mapleleaf'):
        LOCAL_UDF_DATA_DIR = config_json.get_config('REMOTE_DATA_DIR')
    else:
        print 'get_user_data_dir error '
        LOCAL_UDF_DATA_DIR = ''
    print 'your local data dir prefix is: %s' % (LOCAL_UDF_DATA_DIR)
    return LOCAL_UDF_DATA_DIR


override_config = {
    'ENV_TYPE': 'TEST',
    'WORKING_DIR': '/data/home/user00/shield/',
    'SPLIT_SYMBOL': '|',
    'TRANS_SWITCH_NAME': True,
    'TRANS_SWITCH_PINYIN': False,
    'TRANS_SWITCH_PINYIN_WITH_TONE': False,
    'MATCH_SWITCH_SUBSTRING_NAME': True,
    'MATCH_SWITCH_COMMONSTRING': True,
    'WORK_MODE': 'NAMELIST',
    'LOCAL_UDF_DATA_DIR': {
        'shihuanzhao': '/Users/stellazhao/tencent_workplace/labgit/dataming/opsai-monitor/gaojian/test_data/',
        },

    ##PATH CONFIG
    "REMOTE_DATA_DIR": "/data/mapleleaf/shihuan/",

    "original_data": pre + "/order_book_DB",

    "STEP1_DATA_SUBDIR": pre + "/labeled_data/",
    "STEP1_PIC_SUBDIR": pre + "/step1_labeled_data_pic/",

    "STEP2_DATA_SUBDIR": pre + "/step2_model_data",
    "STEP2_PIC_SUBDIR": pre + "/step2_transform_pic",

    "STEP3_DATA_SUBDIR": pre + "/step3_curve_class_data/curve_feature",

    "STEP4_1_DATA_SUBDIR": pre + "/step4_1_humancheck_data",
    "STEP4_1_PIC_SUBDIR": pre + "/step4_1_humancheck_pic",

    "STEP4_2_DATA_COMBINE_XY": pre + "/step4_2_combineXY_data",
    "STEP4_3_MODEL_CLF": pre + "/step4_3_clf_model",

    ##ENV CONFIG
    # "ENV": "REMOTE",
    "ENV": "LOCAL",

    ## ALGORITHM PARAMETERS
    "LABEL_CODE_DICT": {
            "upward": 1,
              "downward": -1,
              "stable": 0
              },

    ########end by shihuanzhao
    'SUM_AYS_WINDOW_LEN': 6,
    'STD_AYS_WINDOW_LEN': 6,
    'BLOCK_AYS_WINDOW_LEN': 3,
    'MAD_WINDOW': 20
}


def get_addition_cfg(biz_id, case_type):
    try:
        res = addition_config[biz_id][case_type]
    except Exception, e:
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




