#!/bin/bash 

# This programme prepares TLG.fst and other graphs for decoding 

dir=$1

if [ $# -lt 1 ]; then
    echo 'Usage: compose_TL.sh <dir>'
    exit 1
fi


for file in L_disambig.fst T.fst; do
    if [ ! -f $dir/$file ]; then
        echo "$dir/$file is missing!" || exit 1;
    fi
done

echo "Preparing graph TL.fst in ${dir}..."

fsttablecompose $dir/T.fst $dir/L_disambig.fst | fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/TL.fst || exit 1;

echo "Done!"
