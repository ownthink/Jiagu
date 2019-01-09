# coding: utf-8
"""
测试 Textrank 模块
"""
import unittest
import jiagu.textrank_zh as t


class TestTextRank(unittest.TestCase):
    def setUp(self):
        self.text = "GNN最近在深度学习领域受到了广泛关注。" \
                    "然而，对于想要快速了解这一领域的研究人员来说，" \
                    "可能会面临着模型复杂、应用门类众多的问题。" \
                    "在内容上，模型方面，本文从GNN原始模型的构建方式" \
                    "与存在的问题出发，介绍了对其进行不同改进的GNN变体，" \
                    "包括如何处理不同的图的类型、如何进行高效的信息传递以" \
                    "及如何加速训练过程。最后介绍了几个近年来提出的通用框架，" \
                    "它们总结概括了多个现有的方法，具有较强的表达能力。"

    def test_text2sentences(self):
        sentences = t.abstract.text2sentences(self.text)
        print(sentences)

    def test_sentences_sort_by_pagerank(self):
        sentences = t.abstract.text2sentences(self.text)
        sentences_sorted = t.abstract.sort_sentences_by_pagerank(sentences)
        print(sentences_sorted)

    def test_text_abstract(self):
        abstract = t.abstract.get_text_abstract(self.text, n=2)
        print(abstract)


if __name__ == '__main__':
    unittest.main()


