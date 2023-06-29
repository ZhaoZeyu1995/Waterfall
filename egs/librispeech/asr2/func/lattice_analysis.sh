#!/bin/bash

# This script is used to analyse the lattice

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./env.sh || exit 1

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

lattice_analysis.sh --nj $nj --acoustic-scale $acoustic_scale --search-beam $search_beam --lattice-beam $lattice_beam $TLG $output_dir
