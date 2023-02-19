#!/bin/bash

. ./path.sh
. ./cmd.sh





nbpe=5000
bpemode=unigram


. utils/parse_options.sh || exit 1;

local_dir=$1
data_dir=$2
dataset_vocab=$3

mkdir -p $local_dir
bpemodel=$local_dir/$(basename $data_dir)_${bpemode}_${nbpe}

cat $data_dir/text | cut -f 2- -d " " > $local_dir/input.txt

(cat $local_dir/input.txt | tr ' ' '\n' | sort | uniq ; cat $dataset_vocab) | sort | uniq > $local_dir/words.txt

spm_train --input=$local_dir/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
spm_encode --model=${bpemodel}.model --output_format=piece < $local_dir/words.txt | tr ' ' '\n' | sort | uniq > $local_dir/nonsilence_phones.txt
spm_encode --model=${bpemodel}.model --output_format=piece < $local_dir/words.txt > $local_dir/lexicon_encoded

echo "<SIL>" > $local_dir/optional_silence.txt
(echo "<UNK>"; echo "<SIL>") > $local_dir/silence_phones.txt

echo "<UNK> <UNK>" > $local_dir/lexicon.txt
echo "<SIL> <SIL>" >> $local_dir/lexicon.txt
paste $local_dir/words.txt $local_dir/lexicon_encoded | awk -F'\t' '{print $1 " " $2}' >> $local_dir/lexicon.txt
