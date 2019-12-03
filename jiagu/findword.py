# -*- encoding:utf-8 -*-
import re
from math import log
from collections import Counter

max_word_len = 6
re_chinese = re.compile(u"[\w]+", re.U)


def count_words(input_file):
    word_freq = Counter()
    fin = open(input_file, 'r', encoding='utf8')
    for index, line in enumerate(fin):
        words = []
        for sentence in re_chinese.findall(line):
            length = len(sentence)
            for i in range(length):
                words += [sentence[i: j + i] for j in range(1, min(length - i + 1, max_word_len + 1))]
        word_freq.update(words)
    fin.close()
    return word_freq


def lrg_info(word_freq, total_word, min_freq, min_mtro):
    l_dict = {}
    r_dict = {}
    for word, freq in word_freq.items():
        if len(word) < 3:
            continue

        left_word = word[:-1]
        right_word = word[1:]

        def __update_dict(side_dict, side_word):
            side_word_freq = word_freq[side_word]
            if side_word_freq > min_freq:
                mul_info1 = side_word_freq * total_word / (word_freq[side_word[1:]] * word_freq[side_word[0]])
                mul_info2 = side_word_freq * total_word / (word_freq[side_word[-1]] * word_freq[side_word[:-1]])
                mul_info = min(mul_info1, mul_info2)
                if mul_info > min_mtro:
                    if side_word in side_dict:
                        side_dict[side_word].append(freq)
                    else:
                        side_dict[side_word] = [side_word_freq, freq]

        __update_dict(l_dict, left_word)
        __update_dict(r_dict, right_word)

    return l_dict, r_dict


def cal_entro(r_dict):
    entro_r_dict = {}
    for word in r_dict:
        m_list = r_dict[word]

        r_list = m_list[1:]

        entro_r = 0
        sum_r_list = sum(r_list)
        for rm in r_list:
            entro_r -= rm / sum_r_list * log(rm / sum_r_list, 2)
        entro_r_dict[word] = entro_r

    return entro_r_dict


def entro_lr_fusion(entro_r_dict, entro_l_dict):
    entro_in_rl_dict = {}
    entro_in_r_dict = {}
    entro_in_l_dict = entro_l_dict.copy()
    for word in entro_r_dict:
        if word in entro_l_dict:
            entro_in_rl_dict[word] = [entro_l_dict[word], entro_r_dict[word]]
            entro_in_l_dict.pop(word)
        else:
            entro_in_r_dict[word] = entro_r_dict[word]
    return entro_in_rl_dict, entro_in_l_dict, entro_in_r_dict


def entro_filter(entro_in_rl_dict, entro_in_l_dict, entro_in_r_dict, word_freq, min_entro):
    entro_dict = {}
    for word in entro_in_rl_dict:
        if entro_in_rl_dict[word][0] > min_entro and entro_in_rl_dict[word][1] > min_entro:
            entro_dict[word] = word_freq[word]

    for word in entro_in_l_dict:
        if entro_in_l_dict[word] > min_entro:
            entro_dict[word] = word_freq[word]

    for word in entro_in_r_dict:
        if entro_in_r_dict[word] > min_entro:
            entro_dict[word] = word_freq[word]

    return entro_dict


def new_word_find(input_file, output_file, min_freq=10, min_mtro=80, min_entro=3):
    word_freq = count_words(input_file)
    total_word = sum(word_freq.values())

    l_dict, r_dict = lrg_info(word_freq, total_word, min_freq, min_mtro)

    entro_r_dict = cal_entro(l_dict)
    entro_l_dict = cal_entro(r_dict)

    entro_in_rl_dict, entro_in_l_dict, entro_in_r_dict = entro_lr_fusion(entro_r_dict, entro_l_dict)
    entro_dict = entro_filter(entro_in_rl_dict, entro_in_l_dict, entro_in_r_dict, word_freq, min_entro)
    result = sorted(entro_dict.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, 'w', encoding='utf-8') as kf:
        for w, m in result:
            kf.write(w + '\t%d\n' % m)
