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
from jiagu import bilstm_crf

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
		if input == 'batch':
			words_list = self.cws_list(sentence)
			return words_list
		else:
			words = self.cws_text(sentence)
			return words
		
	def pos(self, sentence, input='text'):#传入的是词语
		if input == 'batch':
			all_labels = self.pos_model.predict(sentence)
			return all_labels
		else:
			labels = self.pos_model.predict([sentence])[0]
			return labels
			
	def ner(self, sentence, input='text'):#传入的是文本
		if input == 'batch':
			all_labels = self.ner_model.predict(sentence)
			return all_labels
		else:
			labels = self.ner_model.predict([sentence])[0]
			return labels
		
	def cws_pos(self, sentence):
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		text_list = sentence
		all_labels = self.pos_model.predict(text_list)
		return all_labels
	
	def cws_pos_ner(self, sentences):
		'''
		 * seg_pos_ner - 分词、词性标注、命名实体识别
		 * @sentences:    [in]文本列表
		 * 返回词语一一对应的三个列表
		'''
	
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
	
