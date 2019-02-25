# -*- encoding:utf-8 -*-
"""
 * Copyright (C) 2017 OwnThink.
 *
 * Name        : test_findword.py
 * Author      : zengbin93 <zeng_bin8888@163.com>
 * Version     : 0.01
 * Description : 新词发现算法 Unittest
"""

import os
import jiagu
import unittest


class TestFindWord(unittest.TestCase):
    def setUp(self):
        self.input_file = r"C:\迅雷下载\test_msr.txt"
        self.output_file = self.input_file.replace(".txt", '_words.txt')

    def tearDown(self):
        os.remove(self.output_file)

    def test_findword(self):
        jiagu.findword(self.input_file, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))



