# -*- coding: utf-8 -*-
# Standard library
# Third Party Library
# My Library


class order(object):
    def __init__(self, **kwargs):
        self.code = kwargs.get("code", "None")
        self.ask = kwargs.get("ask", -1)
        self.bid = kwargs.get("bid", -1)

    def spread(self):
        return self.bid - self.ask

    def mid_value(self):
        return 0.5 * (self.bid + self.ask)



