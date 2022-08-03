#!/bin/bash


. ./path.sh
. ./cmd.sh

if [ $# != 1 ]; then
  echo "Usage: post_process_and_score.sh <output_dir>"
  echo "e.g.:"
  echo " $0 exp/manual_ctc/version_1/decode_test_dev93"
  exit 1
fi

output_dir=$1

post_process_decode.py $output_dir/hyp.wrd.txt data/$decode_set/utt2spk > $output_dir/hyp.wrd.trn

sclite -r $output_dir/ref.wrd.trn.1 trn -h $output_dir/hyp.wrd.trn trn -i rm -o all stdout > $output_dir/results.wrd.txt
