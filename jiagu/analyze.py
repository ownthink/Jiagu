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
from jiagu import findword
from jiagu import bilstm_crf
from jiagu.textrank import Keywords
from jiagu.textrank import Summarize
from jiagu.segment.nroute import Segment
from jiagu.sentiment.bayes import Bayes
from jiagu.cluster.text import text_cluster as cluster 

def add_curr_dir(name):
	return os.path.join(os.path.dirname(__file__), name)


class Analyze(object):
	def __init__(self):
		self.seg_model = None
		self.pos_model = None
		self.ner_model = None
		
		self.kg_model = None

		self.seg_mmseg = None

		self.keywords_model = None
		self.summarize_model = None
		
		self.seg_nroute = Segment()
		
		self.sentiment_model = Bayes()

	def init(self):
		self.init_cws()
		self.init_pos()
		self.init_ner()
		self.seg_nroute.init()
		
	def load_userdict(self, userdict):
		self.seg_nroute.load_userdict(userdict)

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

	def init_kg(self):
		if self.kg_model is None:
			self.kg_model = bilstm_crf.Predict(add_curr_dir('model/kg.model'))

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

	def seg(self, sentence):
		return self.seg_nroute.seg(sentence, mode="default")
		
	def cws(self, sentence, input='text', model='default'):
		"""中文分词

		:param sentence: str or list
			文本或者文本列表，根据input的模式来定
		:param input: str
			句子输入的格式，text则为默认的文本，batch则为批量的文本列表
		:param model: str
			分词所使用的模式，default为默认模式，mmseg为mmseg分词方式
		:return:
		"""
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

	def knowledge(self, sentence, input='text'):
		self.init_kg()

		if input == 'batch':
			all_labels = self.kg_model.predict(sentence)
			result = []
			for sent, labels in zip(sentence, all_labels):
				result.append(self.lab2spo(sent, labels))
			return result
		else:
			labels = self.kg_model.predict([sentence])[0]
			return self.lab2spo(sentence, labels)	
			
	def keywords(self, text, topkey=5):
		if self.keywords_model == None:
			self.keywords_model = Keywords(tol=0.0001, window=2)
		return self.keywords_model.keywords(text, topkey)

	def summarize(self, text, topsen=5):
		if self.summarize_model == None:
			self.summarize_model = Summarize(tol=0.0001)
		return self.summarize_model.summarize(text, topsen)

	def findword(self, input_file, output_file, min_freq=10, min_mtro=80, min_entro=3):
		findword.new_word_find(input_file, output_file, min_freq, min_mtro, min_entro)
		
	def sentiment(self, text):
		words = self.seg(text)
		ret, prob = self.sentiment_model.classify(words)
		return ret, prob
		
	def text_cluster(self, docs, features_method='tfidf', method="k-means", k=3, max_iter=100, eps=0.5, min_pts=2):
		return cluster(docs, features_method, method, k, max_iter, eps, min_pts, self.seg)
		
	def lab2spo(self, text, epp_labels):
		subject_list = [] # 存放实体的列表
		object_list = []
		index = 0
		for word, ep in zip(list(text), epp_labels):
			if ep[0] == 'B' and ep[2:] == '实体':
				subject_list.append([word, ep[2:], index])
			elif (ep[0] == 'I' or ep[0] == 'E') and ep[2:] == '实体':
				if len(subject_list) == 0:
					continue
				subject_list[len(subject_list)-1][0] += word
			
			if ep[0] == 'B' and ep[2:] != '实体':
				object_list.append([word, ep[2:], index])
			elif (ep[0] == 'I' or ep[0] == 'E') and ep[2:] != '实体':
				if len(object_list) == 0:
					return []
				object_list[len(object_list)-1][0] += word
				
			index += 1
			
		spo_list = []
		if len(subject_list) == 0 or len(object_list) == 0:
			pass
		elif len(subject_list) == 1:
			entity = subject_list[0]
			for obj in object_list:
				predicate = obj[1][:-1] 
				spo_list.append([entity[0], predicate, obj[0]])
		else:
			for obj in object_list:
				entity = []
				predicate = obj[1][:-1]
				direction = obj[1][-1]
				for sub in subject_list:
					if direction == '+':
						if sub[2] > obj[2]:
							entity = sub
							break
					else:
						if sub[2] < obj[2]:
							entity = sub
					
				if entity == []:
					continue
					
				spo_list.append([entity[0], predicate, obj[0]])
				
		return spo_list
		
		