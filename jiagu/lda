# -*- coding: utf-8 -*-

import glob
import jiagu
import numpy as np
from random import random


def normalize(vec):
    total = sum(vec)
    assert(abs(total) > 1e-6)
    for i in range(len(vec)):
        assert(vec[i] >= 0)
        vec[i] = float(vec[i]) / total


def get_prob(vec, prob):
    assert (len(vec) == len(prob))
    # 归一化分布
    normalize(prob)
    r = random()
    index = -1
    while r > 0:
        index = index + 1
        r = r - prob[index]
    return vec[index]


class Document(object):
    def __init__(self, filename):
        self.doc_name = filename[:-4]
        self.__load_document(filename)

    def __load_document(self, filename):
        """
            读取一篇文章，默认一个file里面包含一篇文章
        :param   filename: filename 为 *.txt
        :return: self.document    文章
                 self.words_list  文章中所有的词
        """
        try:
            doc_file = open(filename, "r", encoding="utf-8")
            self.document = ""
            self.words_list = []
            for line in doc_file:
                if line:
                    line = line.strip().replace("\t", "")
                    self.document += line
                    self.words_list.extend(jiagu.seg(line))
        except Exception as e:
            print("无法加载文件，错误信息 : {}".format(e))


class Corpus(object):
    def __init__(self, filepath):
        self.Documents = []
        self.filepath = filepath
        self._build_corpus()

    def _build_corpus(self):
        """
            把所有的文章加载进来
        :return:
        """
        vocabulary = set()
        files = glob.glob(self.filepath + "/*.txt")
        if len(files) > 0:
            for each in files:
                target = Document(each)
                self.Documents.append(target)
                for word in target.words_list:
                    vocabulary.add(word)
            self.vocabulary = list(vocabulary)
            return True
        else:
            print("目标文件夹下没有文件！！！")
            return False


class LdaModel(object):
    def __init__(self, filepath, number_of_topics, alpha=50, beta=0.1, iteration=3):
        self.alpha = alpha
        self.beta = beta
        self.iteration = iteration
        self.corpus = Corpus(filepath)
        self.number_of_topics = number_of_topics
        self.__initialize_all()

    def __initialize_all(self):
        print("LDA Initializing... \nnumber of topics : {}, iteration : {}".format(self.number_of_topics, self.iteration))
        self.number_of_documents = len(self.corpus.Documents)
        assert(self.number_of_documents > self.number_of_topics)
        self.document_topic_counts = np.zeros([self.number_of_documents, self.number_of_topics], dtype=np.int)
        self.topic_word_counts = np.zeros([self.number_of_topics, len(self.corpus.vocabulary)], dtype=np.int)
        self.current_word_topic_assignments = []
        self.topic_counts = np.zeros(self.number_of_topics)
        self.doc_name = dict()
        for d_index, document in enumerate(self.corpus.Documents):
            self.doc_name.setdefault(d_index, document.doc_name)
            word_topic_assignments = []
            for word in document.words_list:
                if word in self.corpus.vocabulary:
                    w_index = self.corpus.vocabulary.index(word)
                    starting_topic_index = np.random.randint(self.number_of_topics)
                    word_topic_assignments.append(starting_topic_index)
                    self.document_topic_counts[d_index, starting_topic_index] += 1
                    self.topic_word_counts[starting_topic_index, w_index] += 1
                    self.topic_counts[starting_topic_index] += 1
            self.current_word_topic_assignments.append(np.array(word_topic_assignments))

        for iteration in range(self.iteration):
            print("Iteration #" + str(iteration + 1) + "...")
            for d_index, document in enumerate(self.corpus.Documents):
                for w, word in enumerate(document.words_list):
                    if word in self.corpus.vocabulary:
                        w_index = self.corpus.vocabulary.index(word)
                        current_topic_index = self.current_word_topic_assignments[d_index][w]
                        self.document_topic_counts[d_index, current_topic_index] -= 1
                        self.topic_word_counts[current_topic_index, w_index] -= 1
                        self.topic_counts[current_topic_index] -= 1
                        topic_distribution = (self.topic_word_counts[:, w_index] + self.beta) * \
                            (self.document_topic_counts[d_index] + self.alpha) / \
                            (self.topic_counts + self.beta)
                        new_topic_index = get_prob(range(self.number_of_topics), topic_distribution)
                        self.current_word_topic_assignments[d_index][w] = new_topic_index
                        self.document_topic_counts[d_index, new_topic_index] += 1
                        self.topic_word_counts[new_topic_index, w_index] += 1
                        self.topic_counts[new_topic_index] += 1
        print("LDA Initializing finished !\n")

    def get_document_topic(self):
        for d_index, topic in enumerate(np.argmax(self.document_topic_counts, axis=1)):
            print("this is file {}, topic : #{}".format(self.doc_name.get(d_index), topic))

    def get_word_topic(self, topN=10):
        for row in (self.topic_word_counts.argsort(axis=1)[:, -topN:]):
            print(list(map(lambda x: self.corpus.vocabulary[x], row)))


if __name__ == "__main__":
    filepath = "documents"
    number_of_topics = 3
    test = LdaModel(filepath, number_of_topics)
    test.get_document_topic()
    test.get_word_topic()

