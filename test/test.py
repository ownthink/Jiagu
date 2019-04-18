import unittest
import jiagu
import sys
import time
import logging

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)


class TestJiagu(unittest.TestCase):
    def test_init(self):
        self.assertIsNone(jiagu.any.seg_model)
        self.assertIsNone(jiagu.any.pos_model)
        self.assertIsNone(jiagu.any.ner_model)
        default_logger.debug('Start initialization.')
        time_start = time.time()
        jiagu.init()  # 可手动初始化，也可以动态初始化
        default_logger.debug(f'Initialization costs {time.time() - time_start:.2f} seconds')
        self.assertIsNotNone(jiagu.any.seg_model)
        self.assertIsNotNone(jiagu.any.pos_model)
        self.assertIsNotNone(jiagu.any.ner_model)

    def test_seg(self, text='厦门明天会不会下雨'):
        words = jiagu.seg(text)  # 默认模式
        print('默认模式分词: ', words)
        self.assertEqual(words, ['厦门', '明天', '会', '不会', '下雨'])
        words = jiagu.seg(text, model="mmseg")  # mmseg模式
        words = list(words)
        print('mmseg模式分词:', words)
        self.assertEqual(words, ['厦门', '明天', '会', '不会', '下雨'])

    def test_pos(self, text='厦门明天会不会下雨'):
        pos = jiagu.pos(text)  # 词性标注
        print('POS tagging result:', [(c, p) for c, p in zip(text, pos)])  # Character-level labeling
        self.assertEqual(len(pos), len(text))
        self.assertEqual(pos, ['n', 'n', 'a', 'nt', 'vu', 'd', 'vu', 'v', 'n'])

    def test_ner(self, text='厦门明天会不会下雨'):
        ner = jiagu.ner(text)  # 命名实体识别
        print('NER result:', [(c, p) for c, p in zip(text, ner)])  # Character-level labeling
        self.assertEqual(len(ner), len(text))
        self.assertEqual(ner, ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])


if __name__ == '__main__':
    unittest.main()
