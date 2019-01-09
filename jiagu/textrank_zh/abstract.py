# coding:utf-8
import math
import numpy as np
import networkx as nx

from jiagu.textrank_zh.utils import AttrDict

# from jiagu import seg, pos

sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…']


def text2sentences(text, sep=None):
    """将文本拆分成句子

    :param text: str
        文本
    :param sep: list
        句子分割符列表，默认 ['?', '!', ';', '？', '！', '。', '；', '……', '…']
    :return: list
    """
    if not sep:
        sep = sentence_sep

    sentences = []
    sentence = []

    for _, char in enumerate(text):
        if char in sep and len(sentence) > 0:
            sentences.append(''.join(sentence))
            sentence = []
        else:
            sentence.append(char)
    return sentences


def sentences_similarity(sentence1, sentence2):
    """默认的用于计算两个句子相似度的函数。

    :param sentence1: list
        分好词的句子
    :param sentence2: list
        分好词的句子
    :return: float
    """
    words = list(set(sentence1 + sentence2))
    vector1 = [float(sentence1.count(word)) for word in words]
    vector2 = [float(sentence2.count(word)) for word in words]

    vector3 = [vector1[x] * vector2[x] for x in range(len(vector1))]
    vector4 = [1 for num in vector3 if num > 0.]
    co_occur_num = sum(vector4)

    if abs(co_occur_num) <= 1e-12:
        return 0.

    denominator = math.log(float(len(sentence1))) + math.log(float(len(sentence2)))  # 分母

    if abs(denominator) < 1e-12:
        return 0.

    return co_occur_num / denominator


def sort_sentences_by_pagerank(sentences,
                               sim_method=sentences_similarity,
                               pg_conf={"alpha": 0.85}):
    """基于 pagerank 对句子进行排序

    :param sentences: list of str
        从 text 中得到的句子列表
    :param sim_method: function
        两个句子之间的相似度计算函数，输入为 sentence1 和 sentence2
    :param pg_conf: dict
        pagerank 算法相关参数
    :return:
    """

    numbers = len(sentences)

    graph = np.zeros((numbers, numbers))

    for si in range(numbers):
        for sj in range(si, numbers):
            sentence1 = list(sentences[si])
            sentence2 = list(sentences[sj])
            sim = sim_method(sentence1, sentence2)
            graph[si, sj] = sim
            graph[sj, si] = sim

    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pg_conf)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    sentences_sorted = []
    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sentences_sorted.append(item)

    return sentences_sorted


def get_text_abstract(text, n=6):
    """基于textrank进行文本摘要

    :param text: str
        文本
    :param n: int
        返回最重要的 n 个句子作为整个文本的摘要
    :return:
    """

    sentences = text2sentences(text, sentence_sep)
    sentences_sorted = sort_sentences_by_pagerank(sentences)
    return sentences_sorted[:n]




