import time
import jiagu

text1 = '你知道国务院吗'
text2 = '厦门明天会不会下雨'
text3 = '火车中将禁止吃东西'
text4 = '“我们不得不同心协力对付这些问题。'

text = [text1, text2, text3, text4]

text = text3

words = jiagu.seg(text)
print(words)

pos = jiagu.pos(words)
print(pos)

ner = jiagu.ner(text)
print(ner)





# words = jiagu.cut(text, input='batch')
# print(words)


