#!/bin/bash


# This programme assumes there is a split$nj dir in $predict_dir
# e.g. apply_prior --suffix soft exp/ctc_LV60K_2l/version_3

. ./path.sh
. ./cmd.sh

nj=10
decode_sets="test_dev93 test_eval92"
train_set="train_si284_sp"
suffix="soft"

. utils/parse_options.sh || exit 1

exp_dir=$1

prior_dir=$exp_dir/align_$train_set/${suffix}_prior.tab

for decode_set in $decode_sets; do
    predict_dir=$exp_dir/decode_$decode_set
    output_dir=$exp_dir/decode_${decode_set}_${suffix}
    run.pl JOB=1:$nj $output_dir/split10/log/apply_prior.JOB.log \
        apply_prior.py --jid JOB --output_dir $output_dir/split${nj} --predict_dir $predict_dir/split${nj} --prior_dir $prior_dir
    cp $predict_dir/ref.wrd.trn.1 $output_dir/ref.wrd.trn.1
done


