#!/usr/bin/env python
# encoding: utf-8
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
            if "value" in node:
                ret.append(node["value"])
        return ret

    def load(self):
        with open(self.trie_file_path, "rb") as f:
            data = pickle.load(f)
        self.root = data


class Chunk:
    def __init__(self, words_list, chrs):
        # self.sentence_sep = ['?', '!', ';', '？', '！', '。', '；', '……', '…', "，", ",", "."]
        self.best_word = words_list[0]
        self.words_num = len(words_list)
        self.length = 0
        self.entropy = 0
        length_square = 0

        for word in words_list:
            word_length = len(word)
            self.length += word_length
            self.entropy += log(chrs.get(word, 1))
            length_square += word_length * word_length

        self.mean = self.length / self.words_num
        self.var = length_square / self.words_num - (self.length / self.words_num) * (self.length / self.words_num)

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
        self.chrs_dic = self._load_freq(filename="data/chars.dic")

    def _load_freq(self, filename):
        chrs_dic = defaultdict()
        with open(add_curr_dir(filename), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    key, value = line.strip().split(" ")
                    chrs_dic.setdefault(key, int(value))
        return chrs_dic

    def __get_start_words(self, sentence):
        if sentence:
            match_words = self.words_dic.get_matches(sentence)
            return match_words if match_words else [sentence[0]]
        else:
            return False

    def __get_chunks(self, sentence):
        # 获取chunk，每个chunk中最多三个词
        first_match_words = self.__get_start_words(sentence)

        for word_one in first_match_words:
            word_one_length = len(word_one)
            second_match_words = self.__get_start_words(sentence[word_one_length:])
            if second_match_words:
                for word_two in second_match_words:
                    word_two_length = len(word_two) + word_one_length
                    third_match_words = self.__get_start_words(sentence[word_two_length:])
                    if third_match_words:
                        for word_three in third_match_words:
                            yield (Chunk([word_one, word_two, word_three], self.chrs_dic))
                    else:
                        yield (Chunk([word_one, word_two], self.chrs_dic))
            else:
                yield (Chunk([word_one], self.chrs_dic))

    def cws(self, sentence):
        """
        :param sentence: 输入的数据
        :return:         返回的分词生成器
        """
        while sentence:
            chunks = self.__get_chunks(sentence)
            word = max(chunks).best_word
            sentence = sentence[len(word):]
            yield word


if __name__ == "__main__":
    mmseg = MMSeg()
    print(list(mmseg.cws("武汉市长江大桥上的日落非常好看，很喜欢看日出日落。")))
    print(list(mmseg.cws("人要是行干一行行一行.")))