#!/bin/bash

# Get output of the neural network by GPUs
# e.g. predict.sh path/to/model


. ./path.sh
. ./env.sh
. ./cmd.sh

gpus=1
batch_size=10
decode_sets="test_yesno"
lang_dir=data/lang_tg

. utils/parse_options.sh

model_dir=$1
exp_dir=$(dirname $(dirname $model_dir)) # this is specific for our exp dir structure

for decode_set in $decode_sets; do
    data_dir=data/$decode_set
    output_dir=$exp_dir/predict_${decode_set}
    mkdir -p $output_dir
    func/predict_gpu_gnnp.sh --gpus $gpus --batch_size $batch_size $data_dir $lang_dir $model_dir $output_dir
done
