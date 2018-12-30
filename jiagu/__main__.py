
import os
import sys
import re
import jiagu

'''
slot_pat = re.compile('#.*?#')


file = sys.argv[0]
fin = open(file, 'r')
fout = open(sys.argv[1], 'w')
for index, line in enumerate(fin):
		line = line.strip()
		if line =='':
				continue
		line = line.replace(' ', '')
		line = ' '.join(chaos.seg(line)[0])
		slots = slot_pat.findall(line)
		for slot in slots:
				line = line.replace(slot, slot.replace(' ', ''))

		line = line.replace('##', '# #')

		fout.write(line+'\n')
		
fin.close()
fout.close()


'''