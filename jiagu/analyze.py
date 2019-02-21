#!/usr/bin/env python3
# -*-coding:utf-8-*-
"""
 * Copyright (C) 2018 OwnThink.
 *
 * Name        : analyze.py - 解析模块
 * Author      : Yener <yener@ownthink.com>
 * Version     : 0.01
 * Description : 
"""
import os
from jiagu import mmseg
from jiagu import textrank
from jiagu import findword
from jiagu import bilstm_crf
from collections import defaultdict
from jiagu.textrank import Keywords
from jiagu.textrank import Summarize


def add_curr_dir(name):
	return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
	def __init__(self):
		self.seg_model = None
		self.pos_model = None
		self.ner_model = None
		
		self.seg_mmseg = None
		
		self.keywords_model = None
		self.summarize_model = None
		
	def init(self):
		self.init_cws()
		self.init_pos()
		self.init_ner()

	def init_cws(self):
		if self.seg_model is None:
			self.seg_model = bilstm_crf.Predict(add_curr_dir('model/cws.model'))

	def load_model(self, model_path):
		self.seg_model = bilstm_crf.Predict(model_path)
	
	def init_pos(self):
		if self.pos_model is None:
			self.pos_model = bilstm_crf.Predict(add_curr_dir('model/pos.model'))

	def init_ner(self):
		if self.ner_model is None:
			self.ner_model = bilstm_crf.Predict(add_curr_dir('model/ner.model'))

	def init_mmseg(self):
		if self.seg_mmseg is None:
			self.seg_mmseg = mmseg.MMSeg()
		
	@staticmethod
	def __lab2word(sentence, labels):
		sen_len = len(sentence)
		tmp_word = ""
		words = []
		for i in range(sen_len):
			label = labels[i]
			w = sentence[i]
			if label == "B":
				tmp_word += w
			elif label == "M":
				tmp_word += w
			elif label == "E":
				tmp_word += w
				words.append(tmp_word)
				tmp_word = ""
			else:
				tmp_word = ""
				words.append(w)
		if tmp_word:
			words.append(tmp_word)
		return words

	def cws_text(self, sentence):
		if sentence == '':
			return ['']
		labels = self.seg_model.predict([sentence])[0]
		return self.__lab2word(sentence, labels)

	def cws_list(self, sentences):
		text_list = sentences
		all_labels = self.seg_model.predict(text_list)
		sent_words = []
		for ti, text in enumerate(text_list):
			seg_labels = all_labels[ti]
			sent_words.append(self.__lab2word(text, seg_labels))
		return sent_words

	def cws(self, sentence, input='text', model='default'):
		'''
		 * cws - 中文分词
		 * @sentence:	[in]文本或者文本列表，根据input的模式来定
		 * @input:		[in]句子输入的格式，text则为默认的文本，batch则为批量的文本列表
		 * @model:		[in]分词所使用的模式，default为默认模式，mmseg为mmseg分词方式
		'''
		if model == 'default':
			self.init_cws()

			if input == 'batch':
				words_list = self.cws_list(sentence)
				return words_list
			else:
				words = self.cws_text(sentence)
				return words
		elif model == 'mmseg':
			self.init_mmseg()
			
			words = self.seg_mmseg.cws(sentence)
			return words
		else:
			pass
		return []

	def pos(self, sentence, input='words'):  # 传入的是词语
		self.init_pos()

		if input == 'batch':
			all_labels = self.pos_model.predict(sentence)
			return all_labels
		else:
			labels = self.pos_model.predict([sentence])[0]
			return labels

	def ner(self, sentence, input='text'):  # 传入的是文本
		self.init_ner()

		if input == 'batch':
			all_labels = self.ner_model.predict(sentence)
			return all_labels
		else:
			labels = self.ner_model.predict([sentence])[0]
			return labels
		
	def keywords(self, text, topkey=5):
		if self.keywords_model == None:
			self.keywords_model = Keywords(tol=0.0001,window=2)
		return self.keywords_model.keywords(text, topkey)
	
	def summarize(self, text, topsen=5):
		if self.summarize_model == None:
			self.summarize_model = Summarize(tol=0.0001)
		return self.summarize_model.summarize(text, topsen)

	def findword(self, input, output):
		findword.train_corpus_words(input, output)
		
		
		