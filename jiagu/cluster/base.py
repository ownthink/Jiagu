# -*-coding:utf-8-*-
import jiagu
from collections import Counter
import numpy as np


def elu_distance(a, b):
    """计算两点之间的欧氏距离并返回"""
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist


def count_features(corpus, tokenizer=jiagu.cut):
    """词频特征

    :param corpus: list of str
    :param tokenizer: function for tokenize, default is `jiagu.cut`
    :return:
        features: np.array
        names: list of str

    example:
    >>> from jiagu.cluster.base import count_features
    >>> corpus = ["判断unicode是否是汉字，数字，英文，或者其他字符。", "全角符号转半角符号。"]
    >>> X, names = count_features(corpus)
    """
    tokens = [tokenizer(x) for x in corpus]
    feature_names = [x[0] for x in Counter([x for s in tokens for x in s]).most_common()]

    features = []
    for sent in tokens:
        counter = Counter(sent)
        feature = [counter.get(x, 0) for x in feature_names]
        features.append(feature)

    return np.array(features), feature_names
