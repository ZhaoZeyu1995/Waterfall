#!/usr/bin/env python3

import sys
import os

'''
By default, nonsilence_phones should not start with '<' as all while silence_phones should start with '<'
nonsilence_phones.py /path/to/phones.txt /path/of/output (including nonsilence_phones.txt silence_phones.txt optional_silence.txt)
By default, optional_silence.txt is '<space>'
'''


def main(phones, dict_dir):

    silence_phones = []
    nonsilence_phones = []
    with open(phones) as f:
        for line in f:
            phone = line.strip().split()[0]
            if phone.startswith('<') and phone != '<unk>':
                silence_phones.append(phone)
            else:
                nonsilence_phones.append(phone)
    with open(os.path.join(dict_dir, 'silence_phones.txt'), 'w') as f:
        f.write('\n'.join(silence_phones))
        f.write('\n')
    with open(os.path.join(dict_dir, 'nonsilence_phones.txt'), 'w') as f:
        f.write('\n'.join(nonsilence_phones))
        f.write('\n')
    with open(os.path.join(dict_dir, 'optional_silence.txt'), 'w') as f:
        f.write('<space>\n')


if __name__ == "__main__":
    phones = sys.argv[1]
    dict_dir = sys.argv[2]
    main(phones, dict_dir)
