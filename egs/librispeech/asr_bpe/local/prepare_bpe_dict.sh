#!/bin/bash
#
# Prepare a BPE dict which is the input for utils/prepare_lang.sh
# This programme generates two dict, for training and testing, respectively.
# data/local/dict_bpe_${nbpe} data/local/dict_bpe_${nbpe}_test
# with different vocabularies for the lexicons.
# For training, the lexicon only contains the words in the training set.
# For testing, the lexicon only contains the words in the testing sets, which may not be completely covered by the LMs, but basically alright.
# Usage: ./local/prepare_bpe_dict.sh ${nbpe} ${bpemode}
# Example: ./local/prepare_bpe_dict.sh 5000 unigram


. ./path.sh || exit 1
. ./cmd.sh || exit 1


nbpe=$1
bpemode=$2

train_set=train_960

train_dict=data/local/dict_bpe_${nbpe}
test_dict=data/local/dict_bpe_${nbpe}_test

[ -d $train_dict ] && rm -rf $train_dict
[ -d $test_dict ] && rm -rf $test_dict

mkdir -p $train_dict
mkdir -p $test_dict

cut -f 2- -d" " data/$train_set/text > $train_dict/input.txt

(cut -f 2- -d" " data/$train_set/text; cut -f 2- -d" " data/dev_clean/text; cut -f 2- -d" " data/dev_other/text)\
    | tr " " "\n" | grep -v "<UNK>" | sort | uniq > $train_dict/words
(cut -f 2- -d" " data/dev_clean/text; cut -f 2- -d" " data/dev_other/text; cut -f 2- -d" " data/test_clean/text; cut -f 2- -d" " data/test_other/text) \
    | tr " " "\n" | grep -v "<UNK>" | sort | uniq > $test_dict/words

bpemodel=$train_dict/${train_set}_${bpemode}_${nbpe}

spm_train --input=$train_dict/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000

(echo "<UNK>"; spm_encode --model=${bpemodel}.model --output_format=piece < $train_dict/words | tr ' ' '\n') | grep -v "<SIL>"| sort | uniq > $train_dict/nonsilence_phones.txt
cp $train_dict/nonsilence_phones.txt $test_dict/nonsilence_phones.txt

spm_encode --model=${bpemodel}.model --output_format=piece < $train_dict/words > $train_dict/lexicon_encoded
spm_encode --model=${bpemodel}.model --output_format=piece < $test_dict/words > $test_dict/lexicon_encoded

echo "<SIL>" > $train_dict/optional_silence.txt
echo "<SIL>" > $train_dict/silence_phones.txt

echo "<SIL>" > $test_dict/optional_silence.txt
echo "<SIL>" > $test_dict/silence_phones.txt

(echo "<SIL> <SIL>"; echo "<UNK> <UNK>"; paste $train_dict/words $train_dict/lexicon_encoded | awk -F'\t' '{print $1 " " $2}') > $train_dict/lexicon.txt
(echo "<SIL> <SIL>"; echo "<UNK> <UNK>"; paste $test_dict/words $test_dict/lexicon_encoded | awk -F'\t' '{print $1 " " $2}') > $test_dict/lexicon.txt
