#!/bin/bash
#
#!/bin/bash
#
# Prepare a phoneme dict, based on the official lexicon in WSJ, which is the input for utils/prepare_lang.sh
# This programme generates two dict, for training and testing, respectively.
# data/local/dict_phone data/local/dict_phone_test
# with different vocabularies for the lexicons.
# For training, the lexicon only contains the words in the training and the dev set.
# For testing, the lexicon only contains the words in the evaluation sets, which may be smaller than the vocabularies in the LMs to reduce the size of the decoding graph.
# Usage: ./local/prepare_phone_dict.sh 
# Example: ./local/prepare_phone_dict.sh
# Note this script needs inputs/lexicon.train_dev.txt and inputs/lexicon.eval.txt which are extracted from the official lexicon in WSJ and CMUdict by Zeyu Zhao.


. ./path.sh || exit 1
. ./cmd.sh || exit 1


train_dict=data/local/dict_phone
test_dict=data/local/dict_phone_test

[ -d $train_dict ] && rm -rf $train_dict
[ -d $test_dict ] && rm -rf $test_dict

mkdir -p $train_dict
mkdir -p $test_dict

(echo "<UNK>"; awk '{$1=""; sub(/^ +/, ""); print $0}' inputs/lexicon.train_dev.txt; awk '{$1=""; sub(/^ +/, ""); print $0}' inputs/lexicon.eval.txt) | tr ' ' '\n' | sort | uniq > $train_dict/nonsilence_phones.txt
cp $train_dict/nonsilence_phones.txt $test_dict/nonsilence_phones.txt

echo "<SIL>" > $train_dict/optional_silence.txt
echo "<SIL>" > $train_dict/silence_phones.txt

echo "<SIL>" > $test_dict/optional_silence.txt
echo "<SIL>" > $test_dict/silence_phones.txt

(echo "<SIL> <SIL>"; echo "<UNK> <UNK>"; cat inputs/lexicon.train_dev.txt) > $train_dict/lexicon.txt
(echo "<SIL> <SIL>"; echo "<UNK> <UNK>"; cat inputs/lexicon.eval.txt) > $test_dict/lexicon.txt
