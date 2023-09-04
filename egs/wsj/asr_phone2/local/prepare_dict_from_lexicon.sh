#!/bin/bash
#
#!/bin/bash
#
# Prepare a phoneme dict from a lexicon.
# Please refer to the inputs/lexicon.train_dev.txt and inputs/lexicon.eval.txt for the format of the lexicon.
# Usage: ./local/prepare_phone_dict.sh <lexicon> <dict_dir>
# Example: ./local/prepare_phone_dict.sh inputs/lexicon.eval.txt data/local/dict_eval


. ./path.sh || exit 1
. ./cmd.sh || exit 1

lexicon=$1
dict_dir=$2

[ -d $dict_dir ] && rm -rf $dict_dir

mkdir -p $dict_dir

(echo "<UNK>"; awk '{$1=""; sub(/^ +/, ""); print $0}' ${lexicon} | tr ' ' '\n' | sort | uniq) > $dict_dir/nonsilence_phones.txt

echo "<SIL>" > $dict_dir/optional_silence.txt
echo "<SIL>" > $dict_dir/silence_phones.txt


(echo "<SIL> <SIL>"; echo "<UNK> <UNK>"; cat ${lexicon}) > $dict_dir/lexicon.txt
