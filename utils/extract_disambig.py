#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith('#'):
            print(line.strip())
