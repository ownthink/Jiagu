# coding: utf-8
from .base import count_features, tfidf_features
from .dbscan import DBSCAN
from .kmeans import KMeans


def text_cluster(docs, features_method='tfidf', method="dbscan",
                 k=3, max_iter=100, eps=0.5, min_pts=2, tokenizer=list):
    """文本聚类，目前支持 K-Means 和 DBSCAN 两种方法

    :param features_method: str
        提取文本特征的方法，目前支持 tfidf 和 count 两种。
    :param docs: list of str
        输入的文本列表，如 ['k-means', 'dbscan']
    :param method: str
        指定使用的方法，默认为 k-means，可选 'k-means', 'dbscan'
    :param k: int
        k-means 参数，类簇数量
    :param max_iter: int
        k-means 参数，最大迭代次数，确保模型不收敛的情况下可以退出循环
    :param eps: float
        dbscan 参数，邻域距离
    :param min_pts:
        dbscan 参数，核心对象中的最少样本数量
    :return: dict
        聚类结果
    """
    if features_method == 'tfidf':
        features, names = tfidf_features(docs, tokenizer)
    elif features_method == 'count':
        features, names = count_features(docs, tokenizer)
    else:
        raise ValueError('features_method error')

    # feature to doc
    f2d = {k: v for k, v in zip(docs, features)}

    if method == 'k-means':
        km = KMeans(k=k, max_iter=max_iter)
        clusters = km.train(features)

    elif method == 'dbscan':
        ds = DBSCAN(eps=eps, min_pts=min_pts)
        clusters = ds.train(features)

    else:
        raise ValueError("method invalid, please use 'k-means' or 'dbscan'")

    clusters_out = {}

    for label, examples in clusters.items():
        c_docs = []
        for example in examples:
            doc = [d for d, f in f2d.items() if list(example) == f]
            c_docs.extend(doc)

        clusters_out[label] = list(set(c_docs))

    return clusters_out
