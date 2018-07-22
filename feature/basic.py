# -*- coding: utf-8 -*-
# Standard library
from math import fabs
# Third Party Library
import numpy as np
from numpy import abs

# My Library


def basic_feature():
    pass


def time_insensitive(P_levels, V_levels):
    """
    :param P_levels:
    :param V_levels:
    :return:
    bid-ask spreads and mid-prices
    price dierences
    mean prices and volumes
    accumulated di↵erences
    """

    N_levels = len(P_levels)
    feature_names = []
    feature_values = []
    for i_level in range(N_levels):
        # v2 feature set:
        feature_names += [
            "spread_level_%s" % (i_level+1), "mid_price_level_%s" % (i_level+1)
        ]
        feature_values += [
            _spread(P_levels[i_level]), _mean(P_levels[i_level])
        ]

        # v3 feature set:
        if i_level >= N_levels - 1:
            continue
        feature_names += ["abs_price_diff_ask_level_%s-%s" % (i_level+2, i_level+1),
                          "abs_price_diff_bid_level_%s-%s" % (i_level+2, i_level+1)]
        feature_value0, feature_value1 = _price_differences(P_levels[i_level + 1], P_levels[i_level])
        feature_values += [abs(feature_value0), abs(feature_value1)]

    feature_names += ["price_diff_ask_level_%s-%s" % (N_levels, 1),
                    "price_diff_bid_level_%s-%s" % (N_levels, 1)]
    feature_value0, feature_value1 = _price_differences(P_levels[-1], P_levels[0])
    feature_values += [feature_value0, -feature_value1]

    # v4 feature set:
    feature_names += [
                "mean_prices_ask",
                "mean_prices_bid",
                "mean_volume_ask",
                "mean_volume_bid",
                ]
    feature_values += [np.mean([p.ask for p in P_levels]),
                       np.mean([p.bid for p in P_levels]),
                       np.mean([v.ask for v in V_levels]),
                       np.mean([v.bid for v in V_levels]),
                       ]
    return feature_names, feature_values


def time_sensitive(P_T1_levels, P_T2_levels,
                    V_T1_levels, V_T2_levels, delta_T):
    # price derivatives
    def derivatives_level(P_t1, P_t2, delta_T):
        derivatives_price_ask = _derivatives(P_t1.ask, P_t2.ask, 0, delta_T)
        derivatives_price_bid = _derivatives(P_t1.bid, P_t2.bid, 0, delta_T)
        return derivatives_price_ask, derivatives_price_bid

    def intensity(p, V):
        return P

    N_levels = len(P_T2_levels)
    feature_names = []
    feature_values = []
    for i_level in range(N_levels):
        # v6 feature sets
        feature_names += [
            "derivatives_price_ask_level_%s" % (i_level + 1),
            "derivatives_price_bid_level_%s" % (i_level + 1),

            "derivatives_volume_ask_level_%s" % (i_level + 1),
            "derivatives_volume_bid_level_%s" % (i_level + 1),
        ]
        feature_values += [
            derivatives_level(P_T1_levels[i_level].ask, P_T2_levels[i_level].ask, delta_T),
            derivatives_level(P_T1_levels[i_level].bid, P_T2_levels[i_level].bid, delta_T),

            derivatives_level(V_T1_levels[i_level].ask, V_T2_levels[i_level].ask, delta_T),
            derivatives_level(V_T1_levels[i_level].bid, V_T2_levels[i_level].bid, delta_T)

        ]


def _derivatives(y1, y2, t1, t2):
    return (y2 - y1) / (t2 - t1)


def _spread(P):
    return P.bid - P.ask


def _mean(P):
    return 0.5 * (P.bid + P.ask)


def _price_differences(P1, P2):
    return (P1.ask - P2.ask,  P1.bid - P2.bid)


if __name__ == "__main__":
    from utils.order import order
    P = order(code="1111.sz", bid=23, ask=21)
    V = order(code="1111.sz", bid=10, ask=5)