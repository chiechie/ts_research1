# -*- coding: utf-8 -*-
# Standard library
# Third Party Library
import pandas as pd

# My Library


useless_col = ["date", "time"]
def delete_useless_column(df):
    for col in df.columns:
        if col in useless_col:
            del df[col]
    return df


# usecols = [
#     u'time', u'date', u'cancel_ask_amount', u'cancel_bid_amount',
#     u'b1_price', u'b2_price', u'b3_price',
#     u'b4_price', u'b5_price',
#     u'b6_price', u'b7_price', u'b8_price',
#     u'b9_price', u'b10_price',
#     u'b1_volume', u'b2_volume', u'b3_volume',
#     u'b4_volume', u'b5_volume',
#     u'b6_volume', u'b7_volume', u'b8_volume',
#     u'b9_volume', u'b10_volume',
#     u'a1_price', u'a2_price', u'a3_price',
#     u'a4_price', u'a5_price',
#     u'a6_price', u'a7_price', u'a8_price',
#     u'a9_price', u'a10_price',
#     u'a1_volume', u'a2_volume', u'a3_volume',
#     u'a4_volume', u'a5_volume',
#     u'a6_volume', u'a7_volume', u'a8_volume',
#     u'a9_volume', u'a10_volume']


def load_level_df(test_path):
    df = pd.read_csv(test_path)
    df.rename(columns={"bid_price": "b1_price"}, inplace=True)
    df = df[df["time"] >= "09:25:00"]
    df["timestamp"] = df["date"] + " " + df["time"]
    df = delete_useless_column(df)
    df["timestamp"] = df["timestamp"].map(pd.to_datetime)
    # df = df.set_index("timestamp")
    # df = df.resample("3S").asfreq()
    # df.fillna(-1, inplace=True)
    return df



