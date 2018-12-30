#!/usr/bin/env python
# -*-coding:utf-8-*-
from jiagu import analyze

any = analyze.Analyze()

initialize = any.initialize

seg = any.seg
pos = any.pos
ner = any.ner
seg_pos = any.seg_pos
seg_pos_ner = any.seg_pos_ner

