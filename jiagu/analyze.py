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
		
		self.init_all = False
		
	def initialize(self):
		print('Jiagu initialize...')
		self.seg_model = bilstm_crf.Predict(add_curr_dir('model/seg.model'))
		self.pos_model = bilstm_crf.Predict(add_curr_dir('model/pos.model'))
		self.ner_model = bilstm_crf.Predict(add_curr_dir('model/ner.model'))
		self.init_all = True
	
	def input_list(self, sentence):
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		if type(sentence) == str:
			text_list = [sentence]
		elif type(sentence) == list:
			text_list = sentence
		else:
			text_list= None
			print('input type error...')
		return text_list
		
	def seg(self, sentence):#传入的是文本
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		if self.init_all == False:
			self.initialize()
			
		text_list = self.input_list(sentence)
		all_labels = self.seg_model.predict(text_list)
		sent_words = []
		for ti, text in enumerate(text_list):
			words = []
			N = len(text)
			seg_labels = all_labels[ti]
			tmp_word = ""
			for i in range(N):
				label = seg_labels[i]
				w = text[i]
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
			sent_words.append(words)
		return sent_words
		
	def pos(self, sentence):#传入的是词语
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		text_list = self.input_list(sentence)
		all_labels = self.pos_model.predict(text_list)
		return all_labels
	
	def ner(self, sentence):#传入的是文本
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		text_list = self.input_list(sentence)
		all_labels = self.ner_model.predict(text_list)
		return all_labels
	
	def seg_pos(self, sentence):
		'''
		 * mention2entity - 提及->实体
		 * @mention:    [in]提及
		 * 根据提及获取歧义关系
		'''
		text_list = self.input_list(sentence)
		all_labels = self.pos_model.predict(text_list)
		return all_labels
	
	def seg_pos_ner(self, sentences):
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
	
