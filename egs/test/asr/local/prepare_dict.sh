#!/bin/bash
#

dir=data/local/dict
[ -d $dir ] && rm -rf $dir
mkdir -p $dir

echo "a a
ab a b
abc a b c
bc b c
b b
c c
<UNK> <UNK>" > $dir/lexicon.txt

echo "<SIL>" > $dir/silence_phones.txt
echo "<SIL>" > $dir/optional_silence.txt

cat $dir/lexicon.txt | cut -f 2- -d" " | tr " " "\n" | sort | uniq > $dir/nonsilence_phones.txt
