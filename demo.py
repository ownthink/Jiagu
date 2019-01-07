import jiagu

#jiagu.init() # 可手动初始化，也可以动态初始化


text = '厦门明天会不会下雨'

words = jiagu.seg(text) # 分词
print(words)

pos = jiagu.pos(words) # 词性标注
print(pos)

ner = jiagu.ner(text) # 命名实体识别
print(ner)



text = '携手推动民族复兴，实现和平统一目标；探索“两制”台湾方案，丰富和平统一实践；坚持一个中国原则，维护和平统一前景；深化两岸融合发展，夯实和平统一基础；实现同胞心灵契合，增进和平统一认同。在《告台湾同胞书》发表40周年纪念会上，习近平总书记提出的这五个方面重大政策主张，系统阐释了实现国家统一的目标内涵、基本方针、路径模式，深刻指明了今后一个时期对台工作的基本思路、重点任务和前进方向，既有坚定的原则性又有极强的针对性和极大的包容性，展现了非凡的政治勇气和政治智慧。'
words = jiagu.seg(text)

stop_words = ['的', '，', '；', '、']
words = [w for w in words if w not in stop_words] # 去除停用词，符号等

keywords = jiagu.keywords(words) # 关键词抽取

print(keywords)


