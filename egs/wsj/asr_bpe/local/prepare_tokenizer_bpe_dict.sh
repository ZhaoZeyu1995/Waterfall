#!/bin/bash
#
# This code cannot be run as it is just an archived version
#
bpe_local_dict=data/local/lang_bpe
nlsyms=$bpe_local_dict/nlsyms
if [ -d $bpe_local_dict ]; then
    rm -rf $bpe_local_dict
fi
mkdir -p $bpe_local_dict

(cat data/train_si284/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq; cat data/lang_nosp_bd/words.txt | awk '{print $1}') \
    | sort | uniq | grep -v -E '<eps>|<s>|</s>|#0|^#' > $bpe_local_dict/words.txt  || exit 1;
(echo "'APOSTROPHE"; echo "'END-INNER-QUOTE"; echo "'END-QUOTE"; echo "'INNER-QUOTE"; echo "'QUOTE"; echo "'SINGLE-QUOTE") > ${nlsyms}_tmp || exit 1;
cat data/lang_nosp_bd/words.txt \
     | awk '{print $1}' | grep -v -E '<eps>|<s>|</s>|#0' | grep -E '^<|^!|^\(|^"|^\)|^,|^-|^\.|^%|^&|^/|^;|^\?' | sort | uniq >> ${nlsyms}_tmp || exit 1;
cat $bpe_local_dict/words.txt | grep -E '^\{|^\}|^~|^<' >> ${nlsyms}_tmp
cat ${nlsyms}_tmp | sort | uniq > ${nlsyms}

# Prepare texts
#
(cat data/train_si284/text; cat data/test_dev93/text; cat data/test_eval92/text) | cut -f 2- -d" " > $bpe_local_dict/texts

tokenise.py $bpe_local_dict/texts $bpe_local_dict/words.txt $bpe_local_dict --nbpe $nbpe --nlsyms $bpe_local_dict/nlsyms

cat $bpe_local_dict/lexicon.txt | cut -f 2- -d" " | tr " " "\n" | sort | uniq > $bpe_local_dict/nonsilence_phones.txt
echo "<space>" > $bpe_local_dict/silence_phones.txt
echo "<space>" > $bpe_local_dict/optional_silence.txt

utils/prepare_lang.sh --position-dependent-phones false $bpe_local_dict \
                    "<UNK>" ${bpe_local_dict}_tmp data/lang_bpe
