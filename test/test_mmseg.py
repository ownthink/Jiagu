#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 * Copyright (C) 2018.
 *
 * Name        : test_mmseg.py
 * Author      : Leo <1162441289@qq.com>
 * Version     : 0.01
 * Description : mmseg分词方法测试
"""

import unittest
import jiagu


class TestTextRank(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_seg_one(self):
        sentence = "人要是行干一行行一行"
        words = jiagu.seg(sentence, model="mmseg")
        self.assertTrue(list(words) == ['人', '要是', '行', '干一行', '行', '一行'])

    def test_seg_two(self):
        sentence = "武汉市长江大桥上的日落非常好看，很喜欢看日出日落。"
        words = jiagu.seg(sentence, model="mmseg")
        self.assertTrue(list(words) == ['武汉市', '长江大桥', '上', '的', '日落', '非常', '好看', '，', '很', '喜欢', '看', '日出日落', '。'])


if __name__ == '__main__':
    unittest.main()
