#!/usr/bin/env python3

# Apache 2.0

import sys

fread = open(sys.argv[1], 'r')

print('0 0 <blk> <eps>')

nodeX = 1
nodes= [] # list of tuple (phone, nodeX). Note each phone corresponds to one state id, nodeX

for entry in fread.readlines():
    entry = entry.replace('\n', '').strip()
    fields = entry.split(' ')
    phone = fields[0]
    if phone == '<eps>' or phone == '<blk>':
        continue
    if '#' in phone:
        print(str(0) + ' ' + str(0) + ' ' + phone + ' ' + phone)
    else:
        print(str(0) + ' ' + str(nodeX) + ' ' + phone + ' ' + phone)
        print(str(nodeX) + ' ' + str(nodeX) + ' ' + phone + ' <eps>')
        print(str(nodeX) + ' ' + str(0) + ' ' + '<blk> <eps>')
        nodes.append((phone, nodeX))
    nodeX += 1
for phone, nodeid in nodes:
    for nid in range(1, len(nodes) + 1):
        if nodeid != nid:
            print('%d %d %s %s' % (nid, nodeid, phone, phone))
    pass

for i in range(len(nodes)+1):
    print('%d' % (i))


fread.close()
