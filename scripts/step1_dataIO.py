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
    df.rename(columns={"bid_level": "b1_price"}, inplace=True)
    df["timestamp"] = df["date"] + " " + df["time"]
    ## for morning
    df1 = df[(df["time"] >= "09:30:00") & (df["time"] <= "11:30:00")]
    df1["timestamp"] = df1["timestamp"].map(pd.to_datetime)
    df1 = df1.set_index("timestamp")
    df1 = df1.resample("3S").bfill()
    print(df1.head())

    ## for afternoon
    df2 = df[(df["time"] >= "14:00:00") & (df["time"] <= "16:00:00")]
    df2["timestamp"] = df2["timestamp"].map(pd.to_datetime)
    df2 = df2.set_index("timestamp")
    df2 = df2.resample("3S").bfill()
    # df.fillna(-1, inplace=True)
    res_df = pd.concat([df1, df2], axis=0)
    res_df = delete_useless_column(res_df)
    print(res_df.columns)
    return res_df



