#!/bin/bash 

. ./path.sh

arpa=/disk/scratch3/zzhao/data/wsj/wsj.arpa
local_dict=data/local/dict

. utils/parse_options.sh || exit 1;


read1gram.py ${arpa} > ${local_dict}/words.txt

words_phones2lexicon.py ${local_dict}/words.txt ${local_dict}/phones.txt > ${local_dict}/lexicon.txt

prepare_phones.py ${local_dict}/phones.txt ${local_dict}


