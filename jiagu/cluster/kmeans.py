# -*-coding:utf-8-*-
"""
 * Copyright (C) 2019 OwnThink.
 *
 * Name        : kmeans.py - 聚类
 * Author      : zengbin93 <zeng_bin8888@163.com>
 * Version     : 0.01
 * Description : KMeans 算法实现
"""

import random
from collections import OrderedDict

from .base import elu_distance


class KMeans(object):
    def __init__(self, k, max_iter=100):
        """

        :param k: int
            类簇数量，如 k=5
        :param max_iter: int
            最大迭代次数，避免不收敛的情况出现导致无法退出循环，默认值为 max_iter=100
        """
        self.k = k
        self.max_iter = max_iter

        self.centroids = None   # list
        self.clusters = None    # OrderedDict

    def _update_clusters(self, dataset):
        """
        对dataset中的每个点item, 计算item与centroids中k个中心的距离
        根据最小距离将item加入相应的簇中并返回簇类结果cluster
        """
        clusters = OrderedDict()
        centroids = self.centroids

        k = len(centroids)
        for item in dataset:
            a = item
            flag = -1
            min_dist = float("inf")

            for i in range(k):
                b = centroids[i]
                dist = elu_distance(a, b)
                if dist < min_dist:
                    min_dist = dist
                    flag = i

            if flag not in clusters.keys():
                clusters[flag] = []
            clusters[flag].append(item)

        self.clusters = clusters

    def _mean(self, features):
        res = []
        for i in range(len(features[0])):
            col = [x[i] for x in features]
            res.append(sum(col) / len(col))
        return res

    def _update_centroids(self):
        """根据簇类结果重新计算每个簇的中心，更新 centroids"""
        centroids = []
        for key in self.clusters.keys():
            centroid = self._mean(self.clusters[key])
            centroids.append(centroid)
        self.centroids = centroids

    def _quadratic_sum(self):
        """计算簇内样本与各自中心的距离，累计求和。

        sum_dist刻画簇内样本相似度, sum_dist越小则簇内样本相似度越高
        计算均方误差，该均方误差刻画了簇内样本相似度
        将簇类中各个点与质心的距离累计求和
        """
        centroids = self.centroids
        clusters = self.clusters

        sum_dist = 0.0
        for key in clusters.keys():
            a = centroids[key]
            dist = 0.0
            for item in clusters[key]:
                b = item
                dist += elu_distance(a, b)
            sum_dist += dist
        return sum_dist

    def train(self, X):
        """输入数据，完成 KMeans 聚类

        :param X: list of list
            输入数据特征，[n_samples, n_features]，如：[[0.36, 0.37], [0.483, 0.312]]
        :return: OrderedDict
        """
        # 随机选择 k 个 example 作为初始类簇均值向量
        self.centroids = random.sample(X, self.k)

        self._update_clusters(X)
        current_dist = self._quadratic_sum()
        old_dist = 0
        iter_i = 0

        while abs(current_dist - old_dist) >= 0.00001:
            self._update_centroids()
            self._update_clusters(X)
            old_dist = current_dist
            current_dist = self._quadratic_sum()

            iter_i += 1
            if iter_i > self.max_iter:
                break

        return self.clusters


