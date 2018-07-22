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


def load_level_df(path):
    df = pd.read_csv(path)
    df.rename(columns={"bid_price": "b1_price"}, inplace=True)
    df = df[df["time"] >= "09:25:00"]
    df["timestamp"] = df["date"] + " " + df["time"]
    df = delete_useless_column(df)
    df["timestamp"] = df["timestamp"].map(pd.to_datetime)
    # df = df.set_index("timestamp")
    # df = df.resample("3S").asfreq()
    # df.fillna(-1, inplace=True)
    return df



