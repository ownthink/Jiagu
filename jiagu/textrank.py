# -*- encoding:utf-8 -*-
import sys
from jiagu import utils
from heapq import nlargest
from collections import defaultdict
from itertools import count, product


class Keywords(object):
    def __init__(self,
                 use_stopword=True,
                 stop_words_file=utils.default_stopwords_file(),
                 max_iter=100,
                 tol=0.0001,
                 window=2):
        self.__use_stopword = use_stopword
        self.__max_iter = max_iter
        self.__tol = tol
        self.__window = window
        self.__stop_words = set()
        self.__stop_words_file = utils.default_stopwords_file()
        if stop_words_file:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            with open(self.__stop_words_file, 'r', encoding='utf-8') as f:
                for word in f:
                    self.__stop_words.add(word.strip())

    @staticmethod
    def build_vocab(sents):
        word_index = {}
        index_word = {}
        words_number = 0
        for word_list in sents:
            for word in word_list:
                if word not in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1
        return word_index, index_word, words_number

    @staticmethod
    def create_graph(sents, words_number, word_index, window=2):
        graph = [[0.0 for _ in range(words_number)] for _ in range(words_number)]
        for word_list in sents:
            for w1, w2 in utils.combine(word_list, window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] += 1.0
                    graph[index2][index1] += 1.0
        return graph

    def keywords(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = utils.as_text(text)
        tokens = utils.cut_sentences(text)
        sentences, sents = utils.psegcut_filter_words(tokens,
                                                      self.__stop_words,
                                                      self.__use_stopword)

        word_index, index_word, words_number = self.build_vocab(sents)
        graph = self.create_graph(sents, words_number,
                                  word_index, window=self.__window)
        scores = utils.weight_map_rank(graph, max_iter=self.__max_iter,
                                       tol=self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(min(len(sent_selected), n)):
            sent_index.append(sent_selected[i][1])
        return [index_word[i] for i in sent_index]


class Summarize(object):
    def __init__(self, use_stopword=True,
                 stop_words_file=None,
                 dict_path=None,
                 max_iter=100,
                 tol=0.0001):
        if dict_path:
            raise RuntimeError("True")
        self.__use_stopword = use_stopword
        self.__dict_path = dict_path
        self.__max_iter = max_iter
        self.__tol = tol

        self.__stop_words = set()
        self.__stop_words_file = utils.default_stopwords_file()
        if stop_words_file:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            with open(self.__stop_words_file, 'r', encoding='utf-8') as f:
                for word in f:
                    self.__stop_words.add(word.strip())

    def filter_dictword(self, sents):
        _sents = []
        dele = set()
        for sentence in sents:
            for word in sentence:
                if word not in self.__word2vec:
                    dele.add(word)
            if sentence:
                _sents.append([word for word in sentence if word not in dele])
        return _sents

    def summarize(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = utils.as_text(text)
        tokens = utils.cut_sentences(text)
        sentences, sents = utils.cut_filter_words(tokens, self.__stop_words, self.__use_stopword)

        graph = self.create_graph(sents)
        scores = utils.weight_map_rank(graph, self.__max_iter, self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(min(n, len(sent_selected))):
            sent_index.append(sent_selected[i][1])
        return [sentences[i] for i in sent_index]

    @staticmethod
    def create_graph(word_sent):
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                board[i][j] = utils.sentences_similarity(word_sent[i], word_sent[j])
        return board

    def compute_similarity_by_avg(self, sents_1, sents_2):
        if len(sents_1) == 0 or len(sents_2) == 0:
            return 0.0
        vec1 = self.__word2vec[sents_1[0]]
        for word1 in sents_1[1:]:
            vec1 = vec1 + self.__word2vec[word1]

        vec2 = self.__word2vec[sents_2[0]]
        for word2 in sents_2[1:]:
            vec2 = vec2 + self.__word2vec[word2]

        similarity = utils.cosine_similarity(vec1 / len(sents_1),
                                             vec2 / len(sents_2))
        return similarity


class TextRank:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, start, end, weight=1):
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        out_sum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            out_sum[n] = sum((e[2] for e in out), 0.0)

        sorted_keys = sorted(self.graph.keys())
        for x in range(10):
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / out_sum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        min_rank, max_rank = sys.float_info[0], sys.float_info[3]
        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws
