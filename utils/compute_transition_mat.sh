#!/bin/bash

# Compute transition_mat of CTC for a data dir
# Usage: compute_transition_mat.sh --nj num_job data_dir lang_dir
# e.g, compute_transition_mat.sh --nj 10 data/train data/lang 
# Outputs are written in data_dir/split$num_job

. ./path.sh
. ./cmd.sh

nj=10

. utils/parse_options.sh || exit 1

data_dir=$1
lang_dir=$2
split_dir=$data_dir/split$nj

mkdir -p $split_dir

run.pl JOB=1:$nj $split_dir/log/split_scp/split_scp.JOB.log split_scp.pl -j $nj JOB --one-based $data_dir/text $split_dir/text.JOB

run.pl JOB=1:$nj $split_dir/log/make_transition_mat/make_transition_mat.JOB.log make_transition_mat.py $split_dir/text.JOB $lang_dir JOB

for i in $(seq $nj); do
    cat $split_dir/transition_mat.$i.scp || exit 1
done > $data_dir/transition_mat.scp || exit 1

