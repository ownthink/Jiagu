# -*- encoding:utf-8 -*-
import os
import jiagu
import math


def default_stopwords_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, 'data/stopwords.txt')


sentence_delimiters = ['。', '？', '！', '…']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns',
                     'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']


def as_text(v):
    """生成unicode字符串"""
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def is_text(v):
    return isinstance(v, str)


def cut_sentences(sentence):
    tmp = []
    for ch in sentence:  # 遍历字符串中的每一个字
        tmp.append(ch)
        if ch in sentence_delimiters:
            yield ''.join(tmp)
            tmp = []
    if len(tmp) > 0:    # 如以定界符结尾的文本的文本信息会在循环中返回，无需再次传递
        yield ''.join(tmp)


def cut_filter_words(cutted_sentences, stopwords, use_stopwords=False):
    sentences = []
    sents = []
    for sent in cutted_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in jiagu.seg(sent) if word and word not in stopwords])  # 把句子分成词语
        else:
            sents.append([word for word in jiagu.seg(sent) if word])
    return sentences, sents


def psegcut_filter_words(cutted_sentences, stopwords, use_stopwords=True):
    sents = []
    sentences = []
    for sent in cutted_sentences:
        sentences.append(sent)

        word_list = jiagu.seg(sent)
        word_list = [word for word in word_list if len(word) > 0]
        if use_stopwords:
            word_list = [word.strip() for word in word_list if word.strip() not in stopwords]
        sents.append(word_list)
    return sentences, sents


def weight_map_rank(weight_graph, max_iter, tol):
    # 初始分数设置为0.5
    # 初始化每个句子的分子和老分数
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    denominator = get_degree(weight_graph)

    # 开始迭代
    count = 0
    while different(scores, old_scores, tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        # 计算每个句子的分数
        for i in range(len(weight_graph)):
            scores[i] = get_score(weight_graph, denominator, i)
        count += 1
        if count > max_iter:
            break
    return scores


def get_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def get_score(weight_graph, denominator, i):
    """

	:param weight_graph:
	:param denominator:
	:param i: int
		第i个句子
	:return: float
	"""
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        # [j,i]是指句子j指向句子i
        fraction = weight_graph[j][i] * 1.0
        # 除以j的出度
        added_score += fraction / denominator[j]
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def different(scores, old_scores, tol=0.0001):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:  # 原始是0.0001
            flag = True
            break
    return flag


def combine(word_list, window=2):
    if window < 2:
        window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def sentences_similarity(s1, s2):
    """计算两个句子的相似度

	:param s1: list
	:param s2: list
	:return: float
	"""
    counter = 0
    for sent in s1:
        if sent in s2:
            counter += 1
    if counter == 0:
        return 0
    return counter / (math.log(len(s1) + len(s2)))


# --------------------------------------------------------------------

def is_chinese(uchar):
    """判断一个字符是否是汉字"""
    assert len(uchar) == 1, "uchar 只能是单个字符"
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个字符是否是数字"""
    assert len(uchar) == 1, "uchar 只能是单个字符"
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个字符是否是英文字母"""
    assert len(uchar) == 1, "uchar 只能是单个字符"
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def B2Q(uchar):
    """单字符半角转全角"""
    assert len(uchar) == 1, "uchar 只能是单个字符"
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:
        # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:
        # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """单字符全角转半角"""
    assert len(uchar) == 1, "uchar 只能是单个字符"
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)
