#!/bin/bash 

# This programme prepares TLG.fst and other graphs for decoding 

dir=$1
tmpdir=$2

if [ $# -lt 2 ]; then
    echo 'Usage: make_graph.sh <dir> <tmpdir>'
    exit 1
fi


for file in L_disambig.fst G.fst; do
    if [ ! -f $dir/$file ]; then
        echo "$dir/$file is missing!" || exit 1;
    fi
done

mkdir -p ${tmpdir}

echo "Preparing decoding graph TLG.fst in ${dir}..."
fsttablecompose $dir/L_disambig.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;

fsttablecompose $dir/T.fst $tmpdir/LG.fst | fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/TLG.fst || exit 1;

fsttablecompose $dir/L_disambig.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstrmsymbols $dir/phones/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/LG.fst || exit 1;
echo "Done!"
