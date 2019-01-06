'''
 * Copyright (C) 2017 OwnThink Technologies Inc. 
 *
 * Name        : analyze.py - 解析
 * Author      : Yener <yener@ownthink.com>
 * Version     : 0.01
 * Description : 解析模块
'''
import os
import sys
import json
from collections import defaultdict
from jiagu import bilstm_crf
from jiagu import textrank

def add_curr_dir(name):
	return os.path.join(os.path.dirname(__file__), name)
						
class Analyze():
	def __init__(self):
		self.seg_model = None
		self.pos_model = None
		self.ner_model = None
		self.init_flag = 0
		
	def init(self):
		self.init_cws() #4
		self.init_pos() #2
		self.init_ner() #1
		
	def init_cws(self):
		if self.seg_model == None:
			self.seg_model = bilstm_crf.Predict(add_curr_dir('model/cws.model'))
			self.init_flag += 4
		
	def init_pos(self):
		if self.pos_model == None:
			self.pos_model = bilstm_crf.Predict(add_curr_dir('model/pos.model'))
			self.init_flag += 2
		
	def init_ner(self):
		if self.ner_model == None:
			self.ner_model = bilstm_crf.Predict(add_curr_dir('model/ner.model'))
			self.init_flag += 1
	
	def __lab2word(self, sentence, labels):
		words = []
		N = len(sentence)
		tmp_word = ""
		for i in range(N):
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
		
	def cws(self, sentence, input='text'):#传入的是文本
		if self.init_flag not in [4, 5, 6, 7]:
			self.init_cws()
			
		if input == 'batch':
			words_list = self.cws_list(sentence)
			return words_list
		else:
			words = self.cws_text(sentence)
			return words
		
	def pos(self, sentence, input='text'):#传入的是词语
		if self.init_flag not in [2, 3, 6, 7]:
			self.init_pos()
			
		if input == 'batch':
			all_labels = self.pos_model.predict(sentence)
			return all_labels
		else:
			labels = self.pos_model.predict([sentence])[0]
			return labels
			
	def ner(self, sentence, input='text'):#传入的是文本
		if self.init_flag not in [1, 3, 5, 7]:
			self.init_ner()
			
		if input == 'batch':
			all_labels = self.ner_model.predict(sentence)
			return all_labels
		else:
			labels = self.ner_model.predict([sentence])[0]
			return labels
		
	def cws_pos(self, sentence):
		text_list = sentence
		all_labels = self.pos_model.predict(text_list)
		return all_labels
	
	def cws_pos_ner(self, sentences):
		seg_list = self.seg(sentences)
		pos_list = self.pos(seg_list)
		ner_list = self.ner(sentences)
		
		results = []
		for i in range(len(sentences)):
			words = seg_list[i]
			pos = pos_list[i]
			ner = ner_list[i]
			results.append([words, pos, ner])

		# listword = []
		# for word, net in zip(sentences, ner):
			# if net[0] == 'B':
				# listword.append(word)
			# elif net[0] == 'I' or net[0] == 'E':
				# listword[len(listword)-1]+=word
			# else:
				# listword.append(word)

			# results.append([words, pos, listword])

		return results
	
	def keywords(self, words, topkey=5):
		'''
		关键字抽取
		'''
		g = textrank.TextRank()

		cm = defaultdict(int)
		span = 5
		for i, wp in enumerate(words):
			for j in range(i + 1, i + span):
				if j >= len(words):
					break
				cm[(wp, words[j])] += 1
				
		for terms, w in cm.items():
			g.addEdge(terms[0], terms[1], w)
		
		nodes_rank = g.rank()

		withWeight = False
		if withWeight:
			tags = sorted(nodes_rank.items(), key=operator.itemgetter(1), reverse=True)
		else:
			tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

		return tags[:topkey]
	
	def abstract(self, text):
		pass
		
		
		
	