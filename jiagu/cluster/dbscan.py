# -*-coding:utf-8-*-
"""
 * Copyright (C) 2019 OwnThink.
 *
 * Name        : dbscan.py - 聚类
 * Author      : zengbin93 <zeng_bin8888@163.com>
 * Version     : 0.01
 * Description : DBSCAN 算法实现
"""

import random
from collections import OrderedDict

from .base import elu_distance


class DBSCAN(object):
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts

    def _find_cores(self, X):
        """遍历样本集找出所有核心对象"""
        cores = set()
        for di in X:
            if len([dj for dj in X if elu_distance(di, dj) <= self.eps]) >= self.min_pts:
                cores.add(di)
        return cores

    def train(self, X):
        """输入数据，完成 KMeans 聚类

        :param X: list of tuple
            输入数据特征，[n_samples, n_features]，如：[[0.36, 0.37], [0.483, 0.312]]
        :return: OrderedDict
        """

        # 确定数据集中的全部核心对象集合
        X = [tuple(x) for x in X]
        cores = self._find_cores(X)
        not_visit = set(X)

        k = 0
        clusters = OrderedDict()
        while len(cores):
            not_visit_old = not_visit
            # 随机选取一个核心对象
            core = list(cores)[random.randint(0, len(cores) - 1)]
            not_visit = not_visit - set(core)

            # 查找所有密度可达的样本
            core_deque = [core]
            while len(core_deque):
                coreq = core_deque[0]
                coreq_neighborhood = [di for di in X if elu_distance(di, coreq) <= self.eps]

                # 若coreq为核心对象，则通过求交集方式将其邻域内未被访问过的样本找出
                if len(coreq_neighborhood) >= self.min_pts:
                    intersection = not_visit & set(coreq_neighborhood)
                    core_deque += list(intersection)
                    not_visit = not_visit - intersection

                core_deque.remove(coreq)
            cluster_k = not_visit_old - not_visit
            cores = cores - cluster_k
            clusters[k] = list(cluster_k)
            k += 1

        return clusters
