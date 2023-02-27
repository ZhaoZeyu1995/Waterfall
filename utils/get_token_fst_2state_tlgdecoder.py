#!/usr/bin/env python3

# Apache 2.0


'''
(0) -> (a0:a) -> (a1:<eps>) -> (0)    Note: () represents self-loop, (0) represents the <blk> self-loop

This special topology is used for the token FST in the T+LG decoder.
The main characteristic of this topology is that all the output labels (phone-ids) are only output at the end of the phone (the transitions arriving at the final state).
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
        print(str(0) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' ' + '<eps>') 
        print(str(nodeX) + ' ' + str(0) + ' ' + phone+'_1' + ' ' + phone) 
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + phone+'_0' + ' <eps>') 
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') 
        print(str(nodeX+1) + ' ' + str(nodeX+1) + ' ' + phone+'_1' + ' <eps>') 
        print(str(nodeX+1) + ' ' + str(0) + ' ' + phone+'_1' + ' ' + phone) 
    nodeX += 2

print('0')


fread.close()
