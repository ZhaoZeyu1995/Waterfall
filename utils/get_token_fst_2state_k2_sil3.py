#!/usr/bin/env python3

# Apache 2.0


'''
(0) -> (a0:a) -> (a1:<eps>) -> (0)    Note: () represents self-loop, (0) represents the <blk> self-loop
'''

import sys

fread = open(sys.argv[1], 'r')


nodeX = 1
nodes= [] # list of tuple (phone, nodeX). Note each phone corresponds to one state id, nodeX


for entry in fread.readlines():
    entry = entry.replace('\n', '').strip()
    fields = entry.split(' ')
    phone = fields[0]
    if phone == '<eps>' or phone.startswith('#'):
        continue
    elif phone == '<SIL>':
        print(str(0) + ' ' + str(0) + ' ' + "<SIL>_0"+ ' ' + "<SIL>")
        print(str(0) + ' ' + str(0) + ' ' + "<SIL>_1"+ ' ' + "<SIL>")
        print(str(0) + ' ' + str(0) + ' ' + "<SIL>_2"+ ' ' + "<SIL>")
        print(str(0) + ' ' + str(nodeX) + ' ' + "<SIL>_0"+ ' ' + "<SIL>")
        print(str(0) + ' ' + str(nodeX) + ' ' + "<SIL>_1"+ ' ' + "<SIL>")
        print(str(0) + ' ' + str(nodeX) + ' ' + "<SIL>_2"+ ' ' + "<SIL>")
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + "<SIL>_0"+ ' ' + "<eps>")
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + "<SIL>_1"+ ' ' + "<eps>")
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + "<SIL>_2"+ ' ' + "<eps>")
        print(str(nodeX) + ' ' + str(0) + ' ' + "<SIL>_0"+ ' ' + "<eps>")
        print(str(nodeX) + ' ' + str(0) + ' ' + "<SIL>_1"+ ' ' + "<eps>")
        print(str(nodeX) + ' ' + str(0) + ' ' + "<SIL>_2"+ ' ' + "<eps>")
        nodeX += 1
    else:
        print(str(0) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' ' + phone) 
        print(str(nodeX) + ' ' + str(0) + ' ' + phone+'_1' + ' <eps>') 
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' <eps>') 
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') 
        print(str(nodeX+1) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') 
        print(str(nodeX+1) + ' ' + str(0) + ' ' + phone+'_1' + ' <eps>') 
        nodeX += 2

print('0')


fread.close()
