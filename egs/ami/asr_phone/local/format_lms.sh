#!/usr/bin/env bash
#
# Copyright  2014 Nickolay V. Shmyrev
# Apache 2.0

if [ -f path.sh ]; then . path.sh; fi

lang=$1
lang_4gsmall=$lang"_fgsmall"
lang_4gbig=$lang"_fgbig"

if [ -d $lang_4gsmall ]; then
  echo "$0: Not regenerating $lang_4gsmall as it already exists."
fi
if [ -d $lang_4gbig ]; then
  echo "$0: Not regenerating $lang_4gbig as it already exists."
fi

small_arpa_lm=data/local/local_lm/data/arpa/4gram_small.arpa.gz
big_arpa_lm=data/local/local_lm/data/arpa/4gram_big.arpa.gz

for f in $small_arpa_lm $big_arpa_lm $lang_4gsmall/words.txt; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


set -e

if [ -f $lang_4gsmall/G.fst ] && [ $lang_4gsmall/G.fst -nt $small_arpa_lm ]; then
  echo "$0: not regenerating $lang_4gsmall/G.fst as it already exists and "
  echo ".. is newer than the source LM."
else
  arpa2fst --disambig-symbol="#0" --read-symbol-table=$lang_4gsmall/words.txt \
    "gunzip -c $small_arpa_lm|" $lang_4gsmall/G.fst
  echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
  fstisstochastic $lang_4gsmall/G.fst || true
  utils/validate_lang.pl --skip-determinization-check $lang_4gsmall || exit 1
fi

if [ -f $lang_4gbig/G.fst ] && [ $lang_4gbig/G.fst -nt $big_arpa_lm ]; then
  echo "$0: not regenerating $lang_4gbig/G.fst as it already exists and "
  echo ".. is newer than the source LM."
else
  arpa2fst --disambig-symbol="#0" --read-symbol-table=$lang_4gbig/words.txt \
    "gunzip -c $big_arpa_lm|" $lang_4gbig/G.fst
  echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
  fstisstochastic $lang_4gbig/G.fst || true
  utils/validate_lang.pl --skip-determinization-check $lang_4gbig || exit 1
fi

exit 0;
