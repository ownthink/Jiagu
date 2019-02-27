import jiagu

# jiagu.init() # 可手动初始化，也可以动态初始化


text = '厦门明天会不会下雨'

words = jiagu.seg(text)  # 分词
print(words)

words = jiagu.seg(text, model="mmseg")  # mmseg分词
print(words)

pos = jiagu.pos(words)  # 词性标注
print(pos)

ner = jiagu.ner(text)  # 命名实体识别
print(ner)
