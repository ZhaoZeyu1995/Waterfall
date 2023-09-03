#!/usr/bin/env bash
#
# Copyright  2014 Nickolay V. Shmyrev
# Modified 2023 University of Edinburgh (Zeyu Zhao)
# Apache 2.0


if [ -f path.sh ]; then . ./path.sh; fi

arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz
[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

if [ -d data/lang_eval_test_tgpr ]; then rm -rf data/lang_eval_test_tgpr; fi
cp -r data/lang_eval data/lang_eval_test_tgpr

gunzip -c "$arpa_lm" | arpa2fst --disambig-symbol=#0 \
  --read-symbol-table=data/lang_eval_test_tgpr/words.txt - data/lang_eval_test_tgpr/G.fst


echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_eval_test_tgpr/G.fst

utils/validate_lang.pl data/lang_eval_test_tgpr || exit 1;

#if [ ! -d data/lang_nosp_rescore ]; then

  #big_arpa_lm=db/cantab-TEDLIUM/cantab-TEDLIUM-unpruned.lm4.gz
  #[ ! -f $big_arpa_lm ] && echo No such file $big_arpa_lm && exit 1;

  #utils/build_const_arpa_lm.sh $big_arpa_lm data/lang_nosp_test data/lang_nosp_rescore || exit 1;

#fi

exit 0;
