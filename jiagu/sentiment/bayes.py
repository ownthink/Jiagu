# -*- coding: utf-8 -*-
import os
import sys
import gzip
import marshal
from math import log, exp

class BaseProb(object):
	def __init__(self):
		self.d = {}
		self.total = 0.0
		self.none = 0

	def exists(self, key):
		return key in self.d

	def getsum(self):
		return self.total

	def get(self, key):
		if not self.exists(key):
			return False, self.none
		return True, self.d[key]

	def freq(self, key):
		return float(self.get(key)[1])/self.total

class AddOneProb(BaseProb):
	def __init__(self):
		self.d = {}
		self.total = 0.0
		self.none = 1

	def add(self, key, value):
		self.total += value
		if not self.exists(key):
			self.d[key] = 1
			self.total += 1
		self.d[key] += value

class Bayes(object):
	def __init__(self):
		self.d = {}
		self.total = 0

	def save(self, fname, iszip=True):
		d = {}
		d['total'] = self.total
		d['d'] = {}
		for k, v in self.d.items():
			d['d'][k] = v.__dict__

		if not iszip:
			marshal.dump(d, open(fname, 'wb'))
		else:
			f = gzip.open(fname, 'wb')
			f.write(marshal.dumps(d))
			f.close()

	def load(self, fname, iszip=True):
		if not iszip:
			d = marshal.load(open(fname, 'rb'))
		else:
			try:
				f = gzip.open(fname, 'rb')
				d = marshal.loads(f.read())
			except IOError:
				f = open(fname, 'rb')
				d = marshal.loads(f.read())
			f.close()
		self.total = d['total']
		self.d = {}
		for k, v in d['d'].items():
			self.d[k] = AddOneProb()
			self.d[k].__dict__ = v

	def train(self, data):
		for d in data:
			c = d[0]
			if c not in self.d:
				self.d[c] = AddOneProb()
			for word in d[1]:
				self.d[c].add(word, 1)
		self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys()))

	def classify(self, x):
		if self.d == {}:
			self.load(os.path.join(os.path.dirname(__file__), 'model/sentiment.model'))
			
		tmp = {}
		for k in self.d:
			tmp[k] = log(self.d[k].getsum()) - log(self.total)
			for word in x:
				tmp[k] += log(self.d[k].freq(word))
		ret, prob = 0, 0
		for k in self.d:
			now = 0
			try:
				for otherk in self.d:
					now += exp(tmp[otherk]-tmp[k])
				now = 1/now
			except OverflowError:
				now = 0
			if now > prob:
				ret, prob = k, now
		return (ret, prob)

if __name__=='__main__':
	classifier = Bayes()

	# 预测
	classifier.load('model/1.model')


	import jiagu

	words = jiagu.seg('今天真的开心')
	
	ret, prob = classifier.classify(words)
	print(ret, prob)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	