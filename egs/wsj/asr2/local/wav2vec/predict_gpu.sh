#!/bin/bash

# Get output of the neural network by GPUs
# e.g. predict_gpu.sh --gpus 1 data/dev data/lang exp/ctc/version_0/checkpoints/epoch=26-vald_loss=0.0000.ckpt/ exp/ctc/version_0


. ./path.sh
. ./cmd.sh

gpus=1
batch_size=10

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: predict_gpu.sh [options] <data_dir> <lang_dir> <model_dir> <output_dir>"
  echo "     --gpus                      # default: 1, the number of gpus used for predicting."
  echo "     --batch_size                # default: 10, batch_size."
  echo "e.g.:"
  echo " $0 --gpus 1 data/train data/lang path/to/model path/for/output"
  exit 1
fi

data_dir=$1 
lang_dir=$2
model_dir=$3
output_dir=$4

mkdir -p $output_dir

predict.py --data_dir $data_dir --lang_dir $lang_dir --model_dir $model_dir --output_dir $output_dir --jid 1 --gpus $gpus --batch_size $batch_size > $output_dir/predict_gpu.log

