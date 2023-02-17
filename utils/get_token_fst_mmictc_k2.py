#!/usr/bin/env python3

# Apache 2.0


'''
(0) -> a0:a -> (a1:<eps>) -> (0)    Note: () represents self-loop, (0) represents the self-loop on state 0
          -> - - - - - - > (0)
'''

import sys

fread = open(sys.argv[1], 'r')


nodeX = 1
nodes= [] # list of tuple (phone, nodeX). Note each phone corresponds to one state id, nodeX


print(str(0) + ' ' + str(0) + ' ' + '<blk>' + ' ' + '<eps>') # <blk> self-loop on state 0

for entry in fread.readlines():
    entry = entry.replace('\n', '').strip()
    fields = entry.split(' ')
    phone = fields[0]
    if phone == '<eps>' or phone == '<SIL>' or phone.startswith('#'):
        continue
    else:
        print(str(0) + ' ' + str(0) + ' ' + phone+'_0' + ' ' + phone) # transiting the first state
        print(str(0) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' ' + phone) # transiting the first state
        print(str(nodeX) + ' ' + str(0) + ' ' + phone+'_1' + ' ' + '<eps>') # transiting the first state
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') #  transiting to the second state
        print(str(nodeX+1) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') #  transiting to the second state
        print(str(nodeX+1) + ' ' + str(0) + ' ' + phone+'_1' + ' <eps>') #  transiting to the second state
    nodeX += 2

print('0')


fread.close()
