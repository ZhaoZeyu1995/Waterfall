#!/bin/bash

# Prepare tokens after prepare_lang.sh
# lang/tokens.txt
# lang/disambig.txt
# lang/disambig.int
# lang/tokens_disambig.txt
# lang/T.fst
# This programme also prepares some files for k2 training
# lang/k2/T.fst # topo Fst for training
# lang/k2/tokens.txt # input labels for T.fst -- no <eps>
# lang/k2/phones.txt # output labels for T.fst -- no <blk> and <eps> is assigned to 0

# usage: prepare_k2.sh data/lang

lang=$1

mkdir -p $lang/k2

get_tokens_2state.py $lang/phones.txt | awk '{print $1 " " NR-1}' > $lang/tokens_disambig.txt

cat $lang/tokens_disambig.txt | grep -v "#" > $lang/tokens.txt
cat $lang/tokens_disambig.txt | grep "#" | awk '{print $2}' > $lang/disambig.int

get_token_fst_2state.py $lang/phones.txt |\
    fstcompile --isymbols=$lang/tokens_disambig.txt --osymbols=$lang/phones.txt \
    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $lang/T.fst

# For k2

cat $lang/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $lang/k2/tokens.txt
cat $lang/phones.txt | grep -v "#" | awk '{print $1 " " NR-1}' > $lang/k2/phones.txt # no disambig symbols

get_token_fst_2state.py $lang/k2/phones.txt |\
    fstcompile --isymbols=$lang/tokens.txt --osymbols=$lang/k2/phones.txt |\
    fstrmepsilon | fstprint --isymbols=$lang/tokens.txt --osymbols=$lang/k2/phones.txt |\
    fstcompile --isymbols=$lang/k2/tokens.txt --osymbols=$lang/k2/phones.txt |\
    fstarcsort --sort_type=olabel > $lang/k2/T.fst
