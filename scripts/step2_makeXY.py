# -*- coding: utf-8 -*-
# Standard library
from os.path import join
from os import listdir
# Third Party Library
import pandas as pd
import numpy as np

# My Library
from common.path_helper import split_dir, saveDF
from settings import Config_json, get_user_data_dir
from step1_dataIO import load_level_df

config_json = Config_json()
root_dir = get_user_data_dir()
input_dir = join(root_dir, config_json.get_config("original_data"))
output_dir = join(root_dir, config_json.get_config("STEP1_DATA_SUBDIR"))
#global parameter for model
DELTA_T2_POINTS = config_json.get_config("DELTA_T2_POINTS")
DELTA_T1_POINTS = config_json.get_config("DELTA_T1_POINTS")
###make spread_crossing labels
delta_Events = config_json.get_config("PREDICT_LENGTH_POINTS")


def spread_crossing(a):
    """
    :param P1: price object contains bid/ask
    :param P2: price object contains bid/ask
    :return: spread_crossing of p1, p2,i.e,p2 - p1
    """
    P1_ask, P1_bid, P2_ask, P2_bid = a
    if P2_bid - P1_ask > 0:
        # "upward"
        return 1
    elif P2_ask - P1_bid < 0:
        # "downward"
        return -1
    else:
        # "stable"
        return 0


# useful_columes = ['askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5',
#                   'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5',
#                   'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
#                   'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']

N_LEVELS = 10
ask_price_level_format = "a%s_price"
ask_volume_level_format = "a%s_volume"
bid_price_level_format = "b%s_price"
bid_volume_level_format = "b%s_volume"

bid_ask_spread_format = "spread%s_price"
mid_prices_format = "mid%s_prices"

useful_columns = [ask_price_level_format % level for level in range(1, N_LEVELS+1)] + \
    [ask_volume_level_format % level for level in range(1, N_LEVELS+1)] + \
    [bid_price_level_format % level for level in range(1, N_LEVELS+1)] + \
    [bid_volume_level_format % level for level in range(1, N_LEVELS+1)]


def makeX(dataSet):
    # Features representation
    ##Basic Set
    ###V1: price and volume (10 levels)
    featV1 = dataSet[useful_columns].values
    featV1_column = useful_columns
    assert len(featV1_column) == featV1.shape[1]

    ##Time-insensitive Set
    ###V2: bid-ask spread and mid-prices
    bid_ask_spread = featV1[:, 0:N_LEVELS] - featV1[:, N_LEVELS * 2: N_LEVELS * 3]
    mid_prices = (featV1[:, 0:N_LEVELS] + featV1[:, N_LEVELS * 2:N_LEVELS * 3]) * 0.5
    featV2 = np.column_stack((bid_ask_spread, mid_prices))
    featV2_column = [bid_ask_spread_format % level for level in range(1, N_LEVELS+1)] + \
        [mid_prices_format % level for level in range(1, N_LEVELS+1)]
    assert len(featV2_column) == featV2.shape[1]

    ###V3: price differences
    ask_price_diffN = featV1[:, N_LEVELS - 1] - featV1[:, 0]
    bid_price_diffN = featV1[:, N_LEVELS * 2] - featV1[:, N_LEVELS * 3 - 1]

    ask_price_diff1 = abs(featV1[:, 1:N_LEVELS] - featV1[:, 0:N_LEVELS-1])
    bid_price_diff1 = abs(featV1[:, N_LEVELS * 2 + 1:N_LEVELS * 3] - featV1[:,  N_LEVELS * 2:N_LEVELS * 3 - 1])
    featV3 = np.column_stack((ask_price_diffN, bid_price_diffN, ask_price_diff1, bid_price_diff1))

    ask_price_diff1_format = "a%s_price_diff1"
    bid_price_diff1_format = "b%s_price_diff1"
    featV3_column = ["a_price_diffN", "b_price_diffN"] +\
                     [ask_price_diff1_format % level for level in range(1, N_LEVELS)] + \
                    [bid_price_diff1_format % level for level in range(1, N_LEVELS)]
    assert len(featV3_column) == featV3.shape[1]

    ###V4: mean prices and volumns
    ask_price_mean = np.mean(featV1[:, 0:N_LEVELS], 1)
    bid_price_mean = np.mean(featV1[:, N_LEVELS * 2:N_LEVELS * 3], 1)
    ask_volume_mean = np.mean(featV1[:, N_LEVELS:N_LEVELS * 2], 1)
    bid_volume_mean = np.mean(featV1[:, N_LEVELS * 3:], 1)
    featV4 = np.column_stack([ask_price_mean, bid_price_mean,
                              ask_volume_mean, bid_volume_mean])
    featV4_column = ["a_mean_price", "b_mean_price",
                     "a_mean_volume", "b_mean_volume"]
    assert len(featV4_column) == featV4.shape[1]

    ###V5: accumulated differences
    price_spread_sum = np.sum(featV2[:, 0:N_LEVELS], 1)
    volume_spread_sum = np.sum(featV1[:, N_LEVELS:N_LEVELS * 2] - featV1[:, N_LEVELS * 3:], 1)
    featV5 = np.column_stack([price_spread_sum, volume_spread_sum])
    featV5_column = ["price_spread_sum", "volume_spread_sum"]
    assert len(featV5_column) == featV5.shape[1]

    ##Time-sensitive Set
    ###V6: price and volume derivatives
    ask_price_derive = featV1[1:, 0:N_LEVELS] - featV1[:-1, 0:N_LEVELS]
    bid_price_derive = featV1[1:, N_LEVELS * 2:N_LEVELS * 3] - featV1[:-1, N_LEVELS * 2:N_LEVELS * 3]
    ask_volume_derive = featV1[1:, N_LEVELS:N_LEVELS * 2] - featV1[:-1, N_LEVELS:N_LEVELS * 2]
    bid_volume_derive = featV1[1:, N_LEVELS * 3:] - featV1[:-1, N_LEVELS * 3:]
    #由于差分，少掉一个数据，此处补回
    featV6_tmp = np.column_stack((ask_price_derive, bid_price_derive,
                                  ask_volume_derive, bid_volume_derive))
    featV6 = np.zeros([ask_price_derive.shape[0]+1, featV6_tmp.shape[1]])
    featV6[1:, :] = featV6_tmp

    ask_price_derive_format = "a%s_price_derive"
    bid_price_derive_format = "b%s_price_derive"
    ask_volume_derive_format = "a%s_volume_derive"
    bid_volume_derive_format = "b%s_volume_derive"

    featV6_column = [ask_price_derive_format % level for level in range(0, N_LEVELS)] + \
                    [bid_price_derive_format % level for level in range(0, N_LEVELS)] + \
                    [ask_volume_derive_format % level for level in range(0, N_LEVELS)] + \
                    [bid_volume_derive_format % level for level in range(0, N_LEVELS)]
    print(len(featV6_column), featV6.shape[1])
    assert len(featV6_column) == featV6.shape[1]

    ###V7: average intensity of each type
    cancel_ask_volume = dataSet["cancel_ask_volume"]
    cancel_bid_volume = dataSet["cancel_bid_volume"]

    current_ask_volume = dataSet["current_ask_volume"]
    current_bid_volume = dataSet["current_bid_volume"]
    ask_lambda_cancel = cancel_ask_volume.diff().fillna(-999)
    bid_lambda_cancel = cancel_bid_volume.diff().fillna(-999)

    total_volume = dataSet["total_volume"]
    total_lambda = total_volume.diff().fillna(-999)

    ask_lambda_limit = total_lambda + cancel_ask_volume + current_ask_volume.diff().fillna(-999)
    bid_lambda_limit = total_lambda + cancel_bid_volume + current_bid_volume.diff().fillna(-999)

    featV7 = np.column_stack([ask_lambda_cancel, bid_lambda_cancel,
                              total_lambda, ask_lambda_limit, bid_lambda_limit])

    featV7_column = ["ask_lambda_cancel", "bid_lambda_cancel",
                              "total_lambda", "ask_lambda_limit", "bid_lambda_limit"]
    assert len(featV7_column) == featV7.shape[1]
    print("featV7_shape", featV7.shape)

    ###V8: relative intensity indicators

    ask_lambda_limit_T1 = pd.Series(ask_lambda_limit).rolling(window=DELTA_T1_POINTS).mean().fillna(0)
    ask_lambda_limit_T2 = pd.Series(ask_lambda_limit).rolling(window=DELTA_T2_POINTS).mean().fillna(0)
    lambda_la_index = (ask_lambda_limit_T1 > ask_lambda_limit_T2).astype(int).values

    bid_lambda_limit_T1 = pd.Series(bid_lambda_limit).rolling(window=DELTA_T1_POINTS).mean().fillna(0)
    bid_lambda_limit_T2 = pd.Series(bid_lambda_limit).rolling(window=DELTA_T1_POINTS).mean().fillna(0)
    lambda_lb_index = (bid_lambda_limit_T1 > bid_lambda_limit_T2).astype(int).values

    ask_lambda_market_T1 = pd.Series(total_lambda).rolling(window=DELTA_T1_POINTS).mean().fillna(0)
    ask_lambda_market_T2 = pd.Series(total_lambda).rolling(window=DELTA_T1_POINTS).mean().fillna(0)
    total_lambda_index = (ask_lambda_market_T1 > ask_lambda_market_T2).astype(int).values

    featV8 = np.column_stack((lambda_la_index, lambda_lb_index, total_lambda_index, ))
    featV8_column = ["lambda_la_index", "lambda_lb_index", "lambda_m_index", ]
    assert len(featV8_column) == featV8.shape[1]
    print("featV8_shape", featV8.shape)

    ###V9: accelerations(market/limit)
    la_derive = np.diff(ask_lambda_limit)
    lb_derive = np.diff(bid_lambda_limit)
    m_derive = np.diff(total_lambda)
    featV9 = np.zeros((featV1.shape[0], 3))
    featV9[1:, :] = np.column_stack((la_derive, lb_derive, m_derive))
    featV9_column = ["lambda_la_derive",
                     "lambda_lb_derive",
                     "lambda_total_derive",
                     ]
    assert len(featV9_column) == featV9.shape[1]

    ##combining the features
    feat = np.column_stack((featV1, featV2, featV3, featV4, featV5, featV6, featV7, featV8, featV9))
    feat_name = featV1_column + featV2_column + featV3_column + featV4_column + featV5_column +\
            featV6_column + featV7_column + featV8_column + featV9_column
    return feat, feat_name



def makeY(dataSet):
    featV1 = dataSet[useful_columns].values
    P1_ask_and_bid = featV1[:-delta_Events, [0, N_LEVELS * 2]]
    P2_ask_and_bid = featV1[delta_Events:, [0, N_LEVELS * 2]]
    feat_tmp = np.column_stack((P1_ask_and_bid, P2_ask_and_bid))
    labels_tmp = np.apply_along_axis(spread_crossing, 1, feat_tmp)
    labels = np.zeros(featV1.shape[0])
    labels[:-delta_Events] = labels_tmp
    return labels

if __name__ == "__main__":
    file_names = [join(input_dir, i) for i in listdir(input_dir) if ".csv" in i]
    for path in file_names:
        dataSet = load_level_df(path)
        feature_value, feature_name = makeX(dataSet=dataSet)
        label = makeY(dataSet=dataSet)
        df = pd.DataFrame(feature_value, columns=feature_name)
        df["label"] = label
        df["label"] = df["label"].astype(int)
        _dir, _filename = split_dir(path)
        out_path = join(output_dir, _filename)
        saveDF(df, out_path)