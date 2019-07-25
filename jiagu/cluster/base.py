# -*-coding:utf-8-*-

import numpy as np


def elu_distance(a, b):
    """计算两点之间的欧氏距离并返回"""
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist

