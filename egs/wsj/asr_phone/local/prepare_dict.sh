#!/bin/bash 

. ./path.sh

arpa=path/to/arpa
train_set=path/to/train_set
local_dict=data/local/dict

. utils/parse_options.sh || exit 1;




cut -f 2- -d" " data/$train_set/text | tr " " "\n" | sort | uniq > ${local_dict}/words.train.txt 

cp ${local_dict}/words.train.txt $local_dict/words.tmp.txt 

if [ -f $arpa ]; then
    read1gram.py ${arpa} > ${local_dict}/words.arpa.txt
    cat ${local_dict}/words.arpa.txt | awk '{print $1}' >> ${local_dict}/words.tmp.txt
fi

cat $local_dict/words.tmp.txt | sort | uniq > $local_dict/words.txt

words_phones2lexicon.py ${local_dict}/words.txt ${local_dict}/phones.txt > ${local_dict}/lexicon.txt

prepare_phones.py ${local_dict}/phones.txt ${local_dict}


