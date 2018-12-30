import time
import pixiu

pixiu.initialize()

text1 = '你知道国务院吗'
text2 = '厦门明天会不会下雨'
text3 = '火车中将禁止吃东西'

text = [text1, text2, text3]

#text = 'hello world 你知道国务院吗'

'''
words = pixiu.seg(text)#可以是文本或者字符串列表
print(words)


pos = pixiu.pos(words)#列表加词列表
print(pos)


ner = pixiu.ner(text)#可以是文本或者字符串列表
print(ner)

'''

results = pixiu.seg_pos_ner(text)


for result in results:
	print(result)
