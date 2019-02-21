#!/usr/bin/env python
# encoding: utf-8
"""
 * Copyright (C) 2018 OwnThink Technologies Inc.
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
    def __init__(self, words, chrs):
        # self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…', "，", ",", "."]
        self.words = words
        self.lens_list = map(lambda x: len(x), words)
        self.length = sum(self.lens_list)
        self.mean = float(self.length) / len(words)
        self.var = sum(map(lambda x: (x - self.mean) ** 2, self.lens_list)) / len(self.words)
        self.entropy = sum([log(float(chrs[x])) for x in words if len(x) == 1 and x in chrs])

    def __lt__(self, other):
        return (self.length, self.mean, -self.var, self.entropy) < \
               (other.length, other.mean, -other.var, other.entropy)


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

    def __get_chunks(self, sentence, depth=3):
        ret = []

        # 递归调用
        def __get_chunks_it(sentence, num, segs):
            if (num == 0 or not sentence) and segs:
                ret.append(Chunk(segs, self.chrs_dic))
            else:
                match_word = self.words_dic.get_matches(sentence)
                if not match_word:
                    __get_chunks_it(sentence[1:], num - 1, segs + [sentence[0]])
                for word in match_word:
                    __get_chunks_it(sentence[len(word):], num - 1, segs + [word])

        __get_chunks_it(sentence, depth, [])
        return ret

    def cws(self, sentence):
        '''
         * cws - 中文分词
         * @sentence:	[in]中文句子输入
         * @return:		[out]返回的分词之后的列表
        '''
        final_ret = []
        while sentence:
            chunks = self.__get_chunks(sentence)
            best = max(chunks)
            final_ret.append(best.words[0])
            sentence = sentence[len(best.words[0]):]
        return final_ret


if __name__ == "__main__":
    mmseg = MMSeg()
    print(mmseg.cws("武汉市长江大桥最近已经崩塌了"))
    print(mmseg.cws("人要是行,干一行行一行一行行行行行要是不行干一行不行一行一行不行行行不行"))
