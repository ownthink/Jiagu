import jiagu

#jiagu.init()


text = '厦门明天会不会下雨'

words = jiagu.seg(text)
print(words)

pos = jiagu.pos(words)
print(pos)

ner = jiagu.ner(text)
print(ner)





