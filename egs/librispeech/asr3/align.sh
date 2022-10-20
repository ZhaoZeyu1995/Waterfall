#!/bin/bash

# Get output of the neural network by GPUs
# e.g. align.sh path/to/model


. ./path.sh
. ./cmd.sh

gpus=1
align_sets="data/train_si284_sp"
lang_dir=data/lang_tg

. utils/parse_options.sh

model_dir=$1
exp_dir=$(dirname $(dirname $model_dir)) # this is specific for our exp dir structure

for align_set in $align_sets; do
    data_dir=data/$decode_set
    output_dir=$exp_dir/align_${align_set}
    mkdir -p $output_dir
    local/wav2vec/align_data.sh --gpus $gpus $data_dir $lang_dir $model_dir $output_dir

    # compute soft prior 
    compute_prior.py --data_dir $data_dir --lang_dir $lang_dir --align_scp $output_dir/align.1.scp > $output_dir/soft_prior.tab
    # compute hard prior
    compute_hard_prior.py --data_dir $data_dir --lang_dir $lang_dir --align_scp $output_dir/align.1.scp > $output_dir/hard_prior.tab
done


