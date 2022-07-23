#!/bin/bash 

langdir=$1
lm=$2
tmpdir=$3
dir=$4

. ./utils/parse_options.sh


for file in L_disambig.fst T.fst; do
    if [ ! -f $langdir/$file ]; then
        echo "$langdir/$file is missing!" || exit 1;
    fi
done

if [ -z $lm ]; then
    echo "LM is missing."
fi

if [ -d $dir ]; then
    rm -rf $dir
fi

cp -r $langdir $dir
mkdir -p ${tmpdir}

arpa2fst --disambig-symbol=#0 --read-symbol-table=$dir/words.txt $lm $dir/G.fst


fsttablecompose $dir/L_disambig.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
fsttablecompose $dir/T.fst $tmpdir/LG.fst | fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/TLG.fst || exit 1;
