#!/usr/bin/env bash
#
# Copyright  2023 University of Edinburgh (Author: Zeyu Zhao)
# Apache 2.0

if [ -f path.sh ]; then . path.sh; fi

src=$1
arpa_lm=$2
dest=$3

if [ -d $dest ]; then
    echo "$0: Not regenerating $dest as it already exists."
else
    cp -r $src $dest || exit 1;
fi


for f in $arpa_lm $src/words.txt; do
    [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

set -e

if [ -f $dest/G.fst ] && [ $dest/G.fst -nt $arpa_lm ]; then
    echo "$0: not regenerating $dest/G.fst as it already exists and "
    echo ".. is newer than the source LM."
else
    arpa2fst --disambig-symbol="#0" --read-symbol-table=$src/words.txt \
        $arpa_lm $dest/G.fst || exit 1;
    echo  "$0: Checking how stochastic G is (the first of these numbers should be small):"
    fstisstochastic $dest/G.fst || true
    utils/validate_lang.pl --skip-determinization-check $dest || exit 1
fi

exit 0;
