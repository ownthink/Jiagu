import time
import jiagu

jiagu.initialize()

text1 = '你知道国务院吗'
text2 = '厦门明天会不会下雨'
text3 = '火车中将禁止吃东西'

text = [text1, text2, text3]


words = jiagu.seg(text)#可以是文本或者字符串列表
print(words)


pos = jiagu.pos(words)#列表加词列表
print(pos)


ner = jiagu.ner(text)#可以是文本或者字符串列表
print(ner)
