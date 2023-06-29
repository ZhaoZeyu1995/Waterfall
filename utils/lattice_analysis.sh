#!/bin/bash
#
# This script is used to analyze the lattice data.

echo "$0 $@"  # Print the command line for logging

nj=10
acoustic_scale=1.0
search_beam=20
lattice_beam=6

. ./utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <TLG> <output-dir>"
  echo "e.g.: $0 data/lang/TLG.pt outputs/*/*/decode_test"
  echo "options:"
  echo "  --nj <nj>                     # number of jobs to run parallelly (default: 10)"
  echo "  --acoustic-scale <scale>      # acoustic scale used for lattice (default: 1.0)"
  echo "  --search-beam <beam>          # beam used for lattice (default: 20)"
  echo "  --lattice-beam <beam>         # beam used for lattice (default: 6)"
  echo "  --help                        # print this message and exit"
  exit 1
fi

TLG=$1
output_dir=$2

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
fi

mkdir -p $output_dir/split$nj/log

run.pl JOB=1:$nj $output_dir/split$nj/log/lattice_analysis.JOB.log lattice_analysis.py --TLG $TLG --scp-split-dir $output_dir/split$nj --acoustic-scale $acoustic_scale --search-beam $search_beam --lattice-beam $lattice_beam --nj JOB

for i in `seq 1 $nj`; do
  cat $output_dir/split$nj/WSumInL.SumL.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $output_dir/WSumInL.SumL.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt

for i in `seq 1 $nj`; do
  cat $output_dir/split$nj/WSumInL.WSumInTLG.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $output_dir/WSumInL.WSumInTLG.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt

for i in `seq 1 $nj`; do
  cat $output_dir/split$nj/WBestInL.WSumInL.SearchBeam$search_beam.LatticeBeam$lattice_beam.$i.txt
done > $output_dir/WBestInL.WSumInL.SearchBeam$search_beam.LatticeBeam$lattice_beam.txt

