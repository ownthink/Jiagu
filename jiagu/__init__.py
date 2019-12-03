#!/usr/bin/env python3
# -*-coding:utf-8-*-
from jiagu import analyze

any = analyze.Analyze()

init = any.init

# 分词
seg = any.seg
cws = any.cws
cut = any.cws

# 词性标注
pos = any.pos

# 命名实体识别
ner = any.ner

# 依存句法分析
# parser

# 加载用户字典
load_userdict = any.load_userdict

# 自定义分词模型
load_model = any.load_model

# 关键字抽取
keywords = any.keywords

# 中文摘要
summarize = any.summarize

# 新词发现
findword = any.findword

# 知识图谱
knowledge = any.knowledge

# 情感分析
sentiment = any.sentiment

# 文本聚类
text_cluster = any.text_cluster