#!/usr/bin/env python
# -*-coding:utf-8-*-
from jiagu import analyze
any = analyze.Analyze()

init = any.init

# 分词
seg = any.cws
cws = any.cws
cut = any.cws

# 词性标注
pos = any.pos

# 命名实体识别
ner = any.ner

# 依存句法分析
# parser

# 加载用户字典
# load_userdict

# 自定义分词模型
load_model = any.load_model


# 其他组合
seg_pos = any.cws_pos
seg_pos_ner = any.cws_pos_ner

# 关键字抽取
keywords = any.keywords

# 中文摘要
summarize = any.summarize

# 新词发现
findword = any.findword

