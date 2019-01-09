# coding: utf-8
import math
import networkx as nx
import numpy as np
from pathlib import Path
from collections import Counter

import jiagu


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def sentences_similarity(sentence1, sentence2):
    """默认的用于计算两个句子相似度的函数。

    :param sentence1: list of str
        分好词的句子
    :param sentence2: list of str
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
                               pg_conf=None):
    """基于 pagerank 对句子进行排序

    :param sentences: list of str
        从 text 中得到的句子列表
    :param sim_method: function
        两个句子之间的相似度计算函数，输入为 sentence1 和 sentence2
    :param pg_conf: dict
        pagerank 算法相关参数
    :return:
    """
    if not pg_conf:
        pg_conf = {"alpha": 0.85}

    numbers = len(sentences)

    graph = np.zeros((numbers, numbers))

    for si in range(numbers):
        for sj in range(si, numbers):
            # sentence1 = list(sentences[si])
            sentence1 = jiagu.cut(sentences[si])
            # sentence2 = list(sentences[sj])
            sentence2 = jiagu.cut(sentences[sj])
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


def create_words_edge(words, window_size=2):
    """构造在window下的单词组合，用来构造单词之间的边。

    :param words: list
    :param window_size: int
    :return: generator
    """
    assert window_size >= 2, "window_size 必须大于等于2，否则无法构建edge"
    for x in range(1, window_size):
        if x >= len(words):
            break
        word_list2 = words[x:]
        res = zip(words, word_list2)
        for r in res:
            yield r


def sort_words_by_pagerank(segments, window_size=2, pg_conf=None):
    """使用pagerank对 words进行排序"""
    if pg_conf is None:
        pg_conf = {"alpha": 0.85}

    # 创建词表
    counter = Counter([word for sentence in segments for word in sentence])
    word2id = {w: i for i, w in enumerate(counter.keys())}
    id2word = {v: k for k, v in word2id.items()}
    words_number = len(word2id)

    # 构建图
    graph = np.zeros((words_number, words_number))

    for sentence in segments:
        for w1, w2 in create_words_edge(sentence, window_size):
            index1 = word2id[w1]
            index2 = word2id[w2]
            graph[index1][index2] = 1.0
            graph[index2][index1] = 1.0

    # 使用pagerank计算顶点权重
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pg_conf)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 输出words排序结果
    sorted_words = []
    for index, score in sorted_scores:
        item = AttrDict(word=id2word[index], weight=score)
        sorted_words.append(item)

    return sorted_words


class TextRank(object):
    def __init__(self, text,
                 sentence_sep=None,
                 seg_method=jiagu.cut,
                 stopwords=None):
        """TextRank 算法实现

        :param text: str
        :param sentence_sep: list
            句子分割符列表
        :param seg_method: function
            用于分词的函数
        :param stopwords: str or list
            如果 stopwords 是 str，则表示文件路径；否则就是 stopwords 列表。
        """
        self.text = text
        self.sentence_sep = sentence_sep
        if self.sentence_sep is None:
            self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…']

        self.seg_method = seg_method

        if isinstance(stopwords, str):
            self.stopwords = Path(stopwords).read_text(encoding='utf-8').split('\n')
        else:
            self.stopwords = stopwords

        # 中间结果
        self.sentences = []
        self.segments = []
        self.get_segments()

    def get_segments(self):
        """文本切分： 分句 & 分词"""
        # 句子切分
        sentences = []
        sentence = []
        for _, char in enumerate(self.text):
            if char in self.sentence_sep and len(sentence) > 0:
                sentences.append(''.join(sentence))
                sentence = []
            else:
                sentence.append(char)

        # 分词
        segments = []
        for s in sentences:
            segment = self.seg_method(s)
            segments.append(segment)

        self.sentences = sentences
        self.segments = segments

    def abstract(self, n=6, sim_method=None, pg_conf=None):
        """文本摘要

        :param n: int
            选择最重要的 n 个句子作为摘要
        :param sim_method: function
            句子相似度计算函数
        :param pg_conf: dict
            pagerank 参数
        :return:
        """
        if not sim_method:
            sim_method = sentences_similarity
        if not pg_conf:
            pg_conf = {"alpha": 0.85}

        sentence_sorted = sort_sentences_by_pagerank(self.sentences,
                                                     sim_method, pg_conf)
        return sentence_sorted[:n]

    def key_words(self, n=10, window_size=2, word_min_len=1, pg_conf=None):
        """

        :param n:
        :param window_size:
        :param pg_conf:
        :return:
        """
        if not pg_conf:
            pg_conf = {"alpha": 0.85}
        words_sorted = sort_words_by_pagerank(self.segments, window_size, pg_conf)

        kw = []
        i = 0
        for item in words_sorted:
            if i > n:
                break
            if len(item.word) >= word_min_len:
                kw.append(item)
                i += 1

        return kw

    def key_phrases(self, kw_nums=12, min_occur_nums=2):
        key_words_set = set([item.word for item in self.key_words(n=kw_nums, word_min_len=1)])

        key_phrases_set = set()
        for sentence in self.segments:
            one = []
            for word in sentence:
                if word in key_words_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        key_phrases_set.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) > 1:
                key_phrases_set.add(''.join(one))

        return [phrase for phrase in key_phrases_set
                if self.text.count(phrase) >= min_occur_nums]










