#!/bin/bash

# Get output of the neural network by CPU
# e.g. predict_cpu.sh --gpus 1 dev data/lang exp/ctc/version_0/checkpoints/epoch=26-vald_loss=0.0000.ckpt/ exp/ctc/version_0

. ./path.sh
. ./cmd.sh


nj=10

. utils/parse_options.sh

decode_set=$1
lang_dir=$2
model_dir=$3
exp_dir=$4

data_dir=data/$decode_set
output_dir="$exp_dir/decode_$decode_set/split$nj"

utils/split_data.sh --per-utt $data_dir $nj

run.pl JOB=1:$nj $output_dir/log/predict.JOB.log predict.py --data_dir $data_dir/split10utt/JOB --lang_dir $lang_dir --model_dir $model_dir --output_dir $output_dir --jid JOB
wait

for i in $(seq $nj); do
    cat $output_dir/split$nj/output.$i.scp || exit 1
done > $output_dir/output.1.scp || exit 1

