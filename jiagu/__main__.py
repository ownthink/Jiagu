#!/usr/bin/env python
# -*-coding:utf-8-*-


"""Fool command line interface."""
import sys
import fool
from argparse import ArgumentParser

parser = ArgumentParser(usage="%s -m fool [options] filename" % sys.executable,
                        description="Fool command line interface.",
                        epilog="If no filename specified, use STDIN instead.")

parser.add_argument("-d", "--delimiter", metavar="DELIM", default=' / ',
                    nargs='?', const=' ',
                    help="use DELIM instead of ' / ' for word delimiter; or a space if it is used without DELIM")

parser.add_argument("-p", "--pos", metavar="DELIM", nargs='?', const='_',
                    help="enable POS tagging; if DELIM is specified, use DELIM instead of '_' for POS delimiter")


parser.add_argument("-u", "--user_dict",
                    help="use USER_DICT together with the default dictionary or DICT (if specified)")

parser.add_argument("-b", "--batch_size", default=1, type = int ,help="batch size ")

parser.add_argument("filename", nargs='?', help="input file")

args = parser.parse_args()

delim = args.delimiter
plim = args.pos

batch_zize = args.batch_size

if args.user_dict:
    fool.load_userdict(args.user_dict)

fp = open(args.filename, 'r') if args.filename else sys.stdin
lines = fp.readlines(batch_zize)


while lines:
    lines = [ln.strip("\r\n") for ln in lines]
    if args.pos:
        result_list  = fool.pos_cut(lines)
        for res in result_list:
            out_str = [plim.join(p) for p in res]
            print(delim.join(out_str))
    else:
        result_list = fool.cut(lines)
        for res in result_list:
            print(delim.join(res))
    lines = fp.readlines(batch_zize)

fp.close()