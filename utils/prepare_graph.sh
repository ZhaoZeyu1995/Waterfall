#!/bin/bash 

# This programme prepares T.fst and TLG.fst and other graphs for training and decoding 



topo=

. ./utils/parse_options.sh

if [ -z $topo ]; then
    echo "--topo cannot be blank!"
    exit 1
fi

srcdir=$1
tmpdir=$2
dir=$3

if [ $# -lt 2 ]; then
    echo 'Usage: prepare_graph.sh [--topo ctc|mmictc|2state|2state-1|3state-skip] <srcdir> <tmpdir> [<desdir>]'
    echo 'By default, <desdir>=${srcdir}_${topo}'
    exit 1
fi

if [ $# -eq 2 ]; then
    dir=${srcdir}_${topo}
fi


for file in L_disambig.fst G.fst; do
    if [ ! -f $srcdir/$file ]; then
        echo "$srcdir/$file is missing!" || exit 1;
    fi
done

if [ -d $dir ]; then
    echo "$dir has already existed. Removing it..."
    rm -rf $dir
fi

cp -r $srcdir $dir
mkdir -p ${tmpdir}

echo "Generating T.fst in ${dir}..."
prepare_${topo}.sh $dir || exit 1;
echo "Done!"

echo "Preparing decoding graph TLG.fst in ${dir}..."
fsttablecompose $dir/L_disambig.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;

fsttablecompose $dir/T.fst $tmpdir/LG.fst | fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/TLG.fst || exit 1;

fsttablecompose $dir/L_disambig.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstrmsymbols $dir/phones/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/LG.fst || exit 1;
echo "Done!"
