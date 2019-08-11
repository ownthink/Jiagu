# -*-coding:utf-8-*-
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from jiagu.cluster.kmeans import KMeans
from jiagu.cluster.dbscan import DBSCAN
from jiagu.cluster.text import text_cluster


def load_dataset():
    # 西瓜数据集4.0  编号，密度，含糖率
    # 数据集来源：《机器学习》第九章 周志华教授
    data = '''
    1,0.697,0.460,
    2,0.774,0.376,
    3,0.634,0.264,
    4,0.608,0.318,
    5,0.556,0.215,
    6,0.403,0.237,
    7,0.481,0.149,
    8,0.437,0.211,
    9,0.666,0.091,
    10,0.243,0.267,
    11,0.245,0.057,
    12,0.343,0.099,
    13,0.639,0.161,
    14,0.657,0.198,
    15,0.360,0.370,
    16,0.593,0.042,
    17,0.719,0.103,
    18,0.359,0.188,
    19,0.339,0.241,
    20,0.282,0.257,
    21,0.748,0.232,
    22,0.714,0.346,
    23,0.483,0.312,
    24,0.478,0.437,
    25,0.525,0.369,
    26,0.751,0.489,
    27,0.532,0.472,
    28,0.473,0.376,
    29,0.725,0.445,
    30,0.446,0.459'''

    data_ = data.strip().split(',')
    dataset = [(float(data_[i]), float(data_[i + 1])) for i in range(1, len(data_) - 1, 3)]
    return np.array(dataset)


def show_dataset():
    dataset = load_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("Dataset")
    plt.show()


def load_docs():
    docs = [
        "百度深度学习中文情感分析工具Senta试用及在线测试",
        "情感分析是自然语言处理里面一个热门话题",
        "AI Challenger 2018 文本挖掘类竞赛相关解决方案及代码汇总",
        "深度学习实践：从零开始做电影评论文本情感分析",
        "BERT相关论文、文章和代码资源汇总",
        "将不同长度的句子用BERT预训练模型编码，映射到一个固定长度的向量上",
        "自然语言处理工具包spaCy介绍",
        "现在可以快速测试一下spaCy的相关功能，我们以英文数据为例，spaCy目前主要支持英文和德文"
    ]
    return docs


class TestCluster(unittest.TestCase):
    def test_a_kmeans(self):
        print("=" * 68, '\n')
        print("test k-means ... ")
        X = load_dataset()
        print("shape of X: ", X.shape)

        k = 4
        km = KMeans(k=k, max_iter=100)
        clusters = km.train(X)
        pprint(clusters)
        self.assertEqual(len(clusters), k)
        pprint({k: len(v) for k, v in clusters.items()})
        print("\n\n")

    def test_b_dbscan(self):
        print("=" * 68, '\n')
        print("test dbscan ... ")
        X = load_dataset()
        ds = DBSCAN(eps=0.11, min_pts=5)
        clusters = ds.train(X)
        pprint(clusters)
        self.assertTrue(len(clusters) < len(X))
        # self.assertEqual(len(clusters), 6)
        pprint({k: len(v) for k, v in clusters.items()})

    def test_c_text_cluster_by_kmeans(self):
        print("=" * 68, '\n')
        print("text_cluster_by_kmeans ... ")
        docs = load_docs()
        clusters = text_cluster(docs, features_method='tfidf', method='k-means', k=3, max_iter=100)
        self.assertTrue(len(clusters) == 3)

    def test_c_text_cluster_by_dbscan(self):
        print("=" * 68, '\n')
        print("text_cluster_by_dbscan ... ")
        docs = load_docs()
        clusters = text_cluster(docs, features_method='count', method='dbscan', eps=5, min_pts=1)
        self.assertTrue(len(clusters) == 3)


if __name__ == '__main__':
    unittest.main()

