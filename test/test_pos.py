import jiagu

text = '厦门明天的天气怎么样'

words = jiagu.seg(text)  # 分词
print(words)

pos = jiagu.pos(words)  # 词性标注
print(pos)

