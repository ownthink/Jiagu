#!/usr/bin/env python
# encoding: utf-8
"""
 * Copyright (C) 2018 OwnThink.
 *
 * Name        : mmseg.py
 * Author      : Leo <1162441289@qq.com>
 * Version     : 0.01
 * Description : mmseg分词方法，目前算法比较耗时，仍在优化中
"""
import os
import pickle
from math import log
from collections import defaultdict


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Trie(object):
    def __init__(self):
        self.root = {}
        self.value = "value"
        self.trie_file_path = os.path.join(os.path.dirname(__file__), "data/Trie.pkl")

    def get_matches(self, word):
        ret = []
        node = self.root
        for c in word:
            if c not in node:
                break
            node = node[c]
            if self.value in node:
                ret.append(node[self.value])
        return ret

    def load(self):
        with open(self.trie_file_path, "rb") as f:
            data = pickle.load(f)
        self.root = data


class Chunk:
    def __init__(self, words_list, chrs, word_freq):
        # self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…', "，", ",", "."]
        self.words = words_list
        self.lens_list = map(lambda x: len(x), words_list)
        self.length = sum(self.lens_list)
        self.mean = float(self.length) / len(words_list)
        self.var = sum(map(lambda x: (x - self.mean) ** 2, self.lens_list)) / len(self.words)
        self.entropy = sum([log(float(chrs.get(x, 1))) for x in words_list])
        # 计算词频信息熵
        self.word_entropy = sum([log(float(word_freq.get(x, 1))) for x in words_list])

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.entropy, self.word_entropy) < \
               (other.length, other.mean, -other.var, other.entropy, other.word_entropy)


class MMSeg:
    def __init__(self):
        # 加载词语字典
        trie = Trie()
        trie.load()
        self.words_dic = trie
        # 加载字频字典
        self.chrs_dic = self._load_freq(filename="data/chars.dic")
        # 加载词频字典
        self.word_freq = self._load_freq(filename="data/words.dic")

    def _load_freq(self, filename):
        chrs_dic = defaultdict()
        with open(add_curr_dir(filename), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    key, value = line.strip().split(" ")
                    chrs_dic.setdefault(key, int(value))
        return chrs_dic

    def __get_start_words(self, sentence):
        match_words = self.words_dic.get_matches(sentence)
        if sentence:
            if not match_words:
                return [sentence[0]]
            else:
                return match_words
        else:
            return False

    def __get_chunks(self, sentence):
        # 获取chunk，每个chunk中最多三个词
        ret = []

        def _iter_chunk(sentence, num, tmp_seg_words):
            match_words = self.__get_start_words(sentence)
            if (not match_words or num == 0) and tmp_seg_words:
                ret.append(Chunk(tmp_seg_words, self.chrs_dic, self.word_freq))
            else:
                for word in match_words:
                    _iter_chunk(sentence[len(word):], num - 1, tmp_seg_words + [word])
        _iter_chunk(sentence, num=3, tmp_seg_words=[])

        return ret

    def cws(self, sentence):
        """
        :param sentence: 输入的数据
        :return:         返回的分词生成器
        """
        while sentence:
            chunks = self.__get_chunks(sentence)
            word = max(chunks).words[0]
            sentence = sentence[len(word):]
            yield word


if __name__ == "__main__":
    mmseg = MMSeg()
    print(list(mmseg.cws("武汉市长江大桥上的日落非常好看，很喜欢看日出日落。")))
    print(list(mmseg.cws("人要是行干一行行一行.")))