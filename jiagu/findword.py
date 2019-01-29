import re
from math import log

"""
http://blog.csdn.net/xiaokang06/article/details/50616983
"""

hanzi_re = re.compile(u"[\w]+", re.U)
PHRASE_MAX_LENGTH = 6

def cut_sentence(sentence):
	result = {}
	sentence_length = len(sentence)
	for i in range(sentence_length):
		for j in range(1, min(sentence_length - i+1, PHRASE_MAX_LENGTH + 1)):
			tmp = sentence[i: j + i]
			result[tmp] = result.get(tmp, 0) + 1
	return result

def gen_word_dict(path):
	word_dict = {}
	with open(path,'r',encoding='utf8') as fp:
		for line in fp:
			hanzi_rdd = hanzi_re.findall(line)
			for words in hanzi_rdd:
				raw_phrase_rdd = cut_sentence(words)
				for word in raw_phrase_rdd:

					if word in word_dict:
						word_dict[word] += raw_phrase_rdd[word]
					else:
						word_dict[word] = raw_phrase_rdd[word]
	return word_dict   
	
def gen_lr_dict(word_dict,counts,thr_fq,thr_mtro):
	l_dict = {}
	r_dict = {}
	k = 0
	for word in word_dict:
		k += 1
		if len(word) < 3: 
			continue
		wordl = word[:-1]
		ml = word_dict[wordl]
		if ml > thr_fq:
			wordl_r = wordl[1:]
			wordl_l = wordl[0]
			mul_info1 = ml * counts / (word_dict[wordl_r] * word_dict[wordl_l])
			wordl_r = wordl[-1]
			wordl_l = wordl[:-1]
			mul_info2 = ml * counts / (word_dict[wordl_r] * word_dict[wordl_l])
			mul_info = min(mul_info1, mul_info2)
			if mul_info > thr_mtro:
				if wordl in l_dict:
					l_dict[wordl].append(word_dict[word])
				else:
					l_dict[wordl] = [ml, word_dict[word]]

		wordr = word[1:]
		mr = word_dict[wordr]
		if mr > thr_fq:
		
			wordr_r = wordr[1:]
			wordr_l = wordr[0]
			mul_info1 = mr * counts / (word_dict[wordr_r] * word_dict[wordr_l])
			wordr_r = wordr[-1]
			wordr_l = wordr[:-1]
			mul_info2 = mr * counts / (word_dict[wordr_r] * word_dict[wordr_l])
			mul_info = min(mul_info1, mul_info2)
			
			if mul_info > thr_mtro:    
				if wordr in r_dict:
					r_dict[wordr].append(word_dict[word])
				else:
					r_dict[wordr] = [mr, word_dict[word]]   
	return l_dict,r_dict
 
def cal_entro(r_dict):
	entro_r_dict = {}
	for word in r_dict:
		m_list = r_dict[word]

		r_list = m_list[1:]
		fm = m_list[0]

		entro_r = 0
		krm = fm - sum(r_list)
		if krm > 0:
			entro_r -= 1 / fm * log(1 / fm, 2) * krm 

		for rm in r_list:
			entro_r -= rm / fm * log(rm / fm, 2)
		entro_r_dict[word] = entro_r
		
	return entro_r_dict
	  
def entro_lr_fusion(entro_r_dict,entro_l_dict):
	entro_in_rl_dict = {}
	entro_in_r_dict = {}
	entro_in_l_dict =  entro_l_dict.copy()
	for word in entro_r_dict:
		if word in entro_l_dict:
			entro_in_rl_dict[word] = [entro_l_dict[word], entro_r_dict[word]]
			entro_in_l_dict.pop(word)
		else:
			entro_in_r_dict[word]  = entro_r_dict[word]
	return entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict
   
def entro_filter(entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict,thr_entro):
	entro_dict = {}
	l, r, rl = 0, 0, 0
	for word in entro_in_rl_dict:
		if entro_in_rl_dict[word][0]>thr_entro and entro_in_rl_dict[word][1]>thr_entro:
			entro_dict[word] = word_dict[word]
			rl +=1

	for word in entro_in_l_dict:
		if entro_in_l_dict[word] > thr_entro:
			entro_dict[word] = word_dict[word]
			l += 1

	for word in entro_in_r_dict:
		if entro_in_r_dict[word] > thr_entro:
			entro_dict[word] = word_dict[word]
			r += 1

	return entro_dict

	
def train_corpus_words(path, output):
	thr_fq = 10  # 词频筛选阈值
	thr_mtro = 80  # 互信息筛选阈值
	thr_entro = 3  # 信息熵筛选阈值
	
	# 步骤1：统计文档所有候选词，词频（包括单字）
	word_dict = gen_word_dict(path)  
	counts = sum(word_dict.values())  # 总词频数

	l_dict,r_dict = gen_lr_dict(word_dict,counts,thr_fq,thr_mtro)  # 右边存在单个字的词 的词典，值为右边字的统计（注意两个词典不一定相同，因为，右边不存在字的词不被记录）

	# 步骤3： 计算左右熵，得到词典：{'一个':5.37,'':,...}
	entro_r_dict = cal_entro(l_dict)  # 左边词词典 计算右边熵
	entro_l_dict = cal_entro(r_dict)  # 右边词词典 计算左边熵
	del l_dict,r_dict  # 释放内存


	# 步骤4：左右熵合并，词典：rl={'一个':[5.37,8.2],'':[左熵，右熵],...},r={'我说':5.37,'':右熵,...},l={'还行吧':3.37,'':左熵,...}
	entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict = entro_lr_fusion(entro_r_dict,entro_l_dict)
	del entro_r_dict,entro_l_dict

	# 步骤5： 信息熵筛选
	entro_dict = entro_filter(entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict,thr_entro)
	del entro_in_rl_dict,entro_in_l_dict,entro_in_r_dict,word_dict
	
	# 步骤6：输出最终满足的词，并按词频排序
	result = sorted(entro_dict.items(), key=lambda x:x[1], reverse=True)

	with open(output, 'w',encoding='utf-8') as kf:
		for w, m in result:
			kf.write(w + ' %d\n' % m)
 
if __name__ == "__main__":
	path = 'msr.txt'
	output = 'count.txt'
	train_corpus_words(path, output)
	
