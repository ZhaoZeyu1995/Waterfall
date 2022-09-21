#!/bin/bash

. ./path.sh
. ./cmd.sh

#langdir=data/lang_tg_deter
dir=data/lang_tg
ngram_lm=/disk/scratch3/zzhao/data/wsj/wsj.arpa
tmpdir=data/local/lang_tmp


#fstminimizeencoded data/lang_tg/TLG.fst | fstarcsort --sort_type=ilabel > data/lang_tg/TLG.minimised.fst

#fstarcsort --sort_type=ilabel $dir/L_disambig.fst > $dir/L_disambig.sorted.fst

#fstdeterminizestar --use-log=true $dir/L_disambig.sorted.fst > $dir/L_disambig.sorted.deted.fst

#fstdeterminizestar --use-log=true $dir/T.fst > $dir/T.deted.fst

#fsttablecompose $dir/T.fst $dir/L_disambig.sorted.fst | fstdeterminizestar --use-log=true | \
  #fstminimizeencoded | fstarcsort --sort_type=olabel > $dir/TL.fst || exit 1;
#fsttablecompose $dir/L_disambig.fst $dir/trivialG.fst | fstdeterminizestar --use-log=true | \
  #fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LtrivialG.fst || exit 1;
fstarcsort --sort_type=ilabel $dir/L_disambig.fst > $dir/L_disambig.sorted.fst
fsttablecompose $dir/T.fst $dir/L_disambig.sorted.fst | fstdeterminizestar --use-log=true | \
   fstrmsymbols $dir/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/TL.fst || exit 1;


# k2-based method 

#python3 -m kaldilm \
      #--read-symbol-table="$langdir/words.txt" \
      #--disambig-symbol='#0' \
      #--max-order=3 \
      #$ngram_lm > $langdir/G.fst.txt

#mkgraph.py $langdir

# k2-based method 
