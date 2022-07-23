#!/bin/bash

# Compute transition_mat and transform_mat of CTC for a data dir
# Usage: compute_transition_and_transform_mat.sh --nj num_job data_dir lang_dir
# e.g, compute_transition_and_transform_mat.sh --nj 10 data/train data/lang 
# Outputs are written in data_dir/split$num_job and data_dir/transition_mat.scp and data_dir/transform_mat.scp

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

run.pl JOB=1:$nj $split_dir/log/make_transform_mat/make_transform_mat.JOB.log make_transform_mat.py $split_dir/text.JOB $lang_dir JOB

for i in $(seq $nj); do
    cat $split_dir/transition_mat.$i.scp || exit 1
done > $data_dir/transition_mat.scp || exit 1

for i in $(seq $nj); do
    cat $split_dir/transform_mat.$i.scp || exit 1
done > $data_dir/transform_mat.scp || exit 1
