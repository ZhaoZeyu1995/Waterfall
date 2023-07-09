#!/bin/bash
#
# This script is used to analyze the lattice data.

echo "$0 $@"  # Print the command line for logging

nj=10
acoustic_scale=1.0
search_beam=20.0
lattice_beam=6.0
max_active=2000

. ./utils/parse_options.sh

if [ $# != 5 ]; then
  echo "Usage: $0 [options] <TLG> <output-dir> <utt2spk> <words> <text>"
  echo "e.g.: $0 data/lang/TLG.pt outputs/*/*/decode_test data/test/utt2spk data/lang/words.txt data/test/text"
  echo "options:"
  echo "  --nj <nj>                     # number of jobs to run parallelly (default: 10)"
  echo "  --acoustic-scale <scale>      # acoustic scale used for lattice (default: 1.0)"
  echo "  --max_active <maxac>          # maximum of the number of active states (default: 2000)"
  echo "  --search-beam <beam>          # beam used for lattice (default: 20.0)"
  echo "  --lattice-beam <beam>         # beam used for lattice (default: 6.0)"
  echo "  --help                        # print this message and exit"
  exit 1
fi

TLG=$1
decode_dir=$2
utt2spk=$3
words=$4
text=$5

if [ ! -f $TLG ]; then
    lang_dir=`dirname $TLG`
    # Check if lang_dir ends with mmictc-1
    if [[ $lang_dir == *"mmictc-1" ]]; then
        echo "$0: Using mmictc-1 topology for TLG compilation. Determinization is not required."
        determinize = false
    else
        determinize = true
    fi
    echo "$0: $TLG does not exist."
    echo "$0: Calling compile_TLG.py to compile TLG.pt in $lang_dir"
    compile_TLG.py $lang_dir $determinize || exit 1
else
    echo "$0: $TLG exists."
fi

mkdir -p $decode_dir/split$nj
predict_dir=$(dirname $decode_dir)


run.pl JOB=1:$nj $decode_dir/split${nj}/log/lattice_analysis.JOB.log lattice_analysis.py --TLG $TLG --scp-split-dir $predict_dir/split$nj --output-dir $decode_dir/split${nj} --acoustic-scale $acoustic_scale --search-beam $search_beam --lattice-beam $lattice_beam --max-active $max_active --nj JOB --words $words --text ${text}

for i in `seq 1 $nj`; do
  cat $decode_dir/split$nj/WSumInL.SumL.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $decode_dir/WSumInL.SumL.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt || exit 1

for i in `seq 1 $nj`; do
  cat $decode_dir/split$nj/WSumInL.WSumInTLG.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $decode_dir/WSumInL.WSumInTLG.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt || exit 1

for i in `seq 1 $nj`; do
  cat $decode_dir/split$nj/WBestInL.WSumInL.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $decode_dir/WBestInL.WSumInL.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt || exit 1

for i in `seq 1 $nj`; do
    utils/int2sym.pl -f 2- $words $decode_dir/split$nj/hyp.$i.wrd | cat || exit 1
done > $decode_dir/hyp.wrd.txt || exit 1

post_process_decode.py $decode_dir/hyp.wrd.txt $utt2spk > $decode_dir/hyp.wrd.trn
sclite -r $predict_dir/ref.wrd.trn.1 trn -h $decode_dir/hyp.wrd.trn trn -i rm -o all stdout > $decode_dir/results.wrd.txt

