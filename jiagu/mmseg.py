#!/usr/bin/env python
# encoding: utf-8
"""
@version   : 0.1
@author    : Leo
@contact   : 1162441289@qq.com
@software  : PyCharm 
@file      : mmseg.py
@time      : 2019/1/5 9:55
@intro     : mmseg分词方法
"""
import os
import pickle
from math import log
from collections import defaultdict


def add_curr_dir(name):
    return os.path.join(os.path.dirname(__file__), name)


class Trie(object):
    def __init__(self):
        self.trie_file_path = os.path.join(os.path.dirname(__file__), "data/Trie.pkl")
        self.root = {}
        self.value = "value"

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
    def __init__(self, words, chrs):
        # self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…', "，", ",", "."]
        self.words = words
        self.lens = map(lambda x: len(x), words)
        self.length = sum(self.lens)
        self.mean = float(self.length) / len(words)
        self.var = sum(map(lambda x: (x - self.mean) ** 2, self.lens)) / len(self.words)
        self.degree = sum([log(float(chrs[x])) for x in words if len(x) == 1 and x in chrs])

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.degree) < \
               (other.length, other.mean, -other.var, other.degree)


class MMSeg:
    def __init__(self):
        # 加载词语字典
        trie = Trie()
        trie.load()
        self.words_dic = trie
        # 加载字频字典
        self.chrs_dic = self._load_word_freq()

    def _load_word_freq(self):
        chrs_dic = defaultdict()
        with open(add_curr_dir('data/chars.dic'), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    key, value = line.strip().split(" ")
                    chrs_dic.setdefault(key, int(value))
        return chrs_dic

    def __get_chunks(self, s, depth=3):
        ret = []

        # 递归调用
        def __get_chunks_it(s, num, segs):
            if (num == 0 or not s) and segs:
                ret.append(Chunk(segs, self.chrs_dic))
            else:
                m = self.words_dic.get_matches(s)
                if not m:
                    __get_chunks_it(s[1:], num - 1, segs + [s[0]])
                for w in m:
                    __get_chunks_it(s[len(w):], num - 1, segs + [w])

        __get_chunks_it(s, depth, [])
        return ret

    def cws(self, s):
        final_ret = []
        while s:
            chunks = self.__get_chunks(s)
            best = max(chunks)
            final_ret.append(best.words[0])
            s = s[len(best.words[0]):]
        return final_ret


if __name__ == "__main__":
    mmseg = MMSeg()
    print(mmseg.cws("武汉市长江大桥最近已经崩塌了"))
    print(mmseg.cws("人要是行,干一行行一行一行行行行行要是不行干一行不行一行一行不行行行不行"))
