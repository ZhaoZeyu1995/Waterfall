#!/bin/bash

# This script is used to analyse the lattice

. ./path.sh || exit 1
. ./cmd.sh || exit 1

echo "$0 $@"  # Print the command line for logging

nj=10
acoustic_scale=1.0
search_beam=20.0
lattice_beam=6.0
max_active=2000

. ./utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <predict-dir> <output-dir>"
  echo "e.g.: $0 data/test data/lang_test outputs/*/*/predict_test outputs/*/*/decode_test_lattice"
  echo "options:"
  echo "  --nj <nj>                     # number of jobs to run parallelly (default: 10)"
  echo "  --acoustic-scale <scale>      # acoustic scale used for lattice (default: 1.0)"
  echo "  --search-beam <beam>          # beam used for lattice (default: 20)"
  echo "  --lattice-beam <beam>         # beam used for lattice (default: 6)"
  echo "  --help                        # print this message and exit"
  exit 1
fi


data_dir=$1
lang_dir=$2
predict_dir=$3
output_dir=$4

TLG=${lang_dir}/TLG.pt

mkdir -p $output_dir
cp $predict_dir/ref.wrd.trn.1 $output_dir/ref.wrd.trn.1

# split the output.1.scp
if [ ! -d $output_dir/split$nj ]; then
    if [ -f $predict_dir/output.1.scp ]; then
        run.pl JOB=1:$nj $output_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $predict_dir/output.1.scp $output_dir/split$nj/output.JOB.scp
    else
        echo "Cannot find output.1.scp or split$nj in $predict_dir!!! There should be one of them at least!"
    fi
fi

decode_dir=$output_dir/acwt_${acoustic_scale}-beam_${search_beam}-latbeam_${lattice_beam}-maxac_${max_active}
mkdir -p $decode_dir
lattice_analysis.sh --nj $nj --acoustic-scale $acoustic_scale --search-beam $search_beam --lattice-beam $lattice_beam --max_active ${max_active} $TLG $decode_dir $data_dir/utt2spk $lang_dir/words.txt $data_dir/text
