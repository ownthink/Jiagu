
import os
import sys
import re
import chaos

# from argparse import ArgumentParser
# parser = ArgumentParser(usage="%s -m jieba [options] filename" % sys.executable, description="Jieba command line interface.", epilog="If no filename specified, use STDIN instead.")
# parser.add_argument("filename", nargs='?', help="input file")
# args = parser.parse_args()

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


