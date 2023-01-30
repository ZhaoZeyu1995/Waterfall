#!/bin/bash

# Get alignments for data given a model on GPUs. 
# e.g. align.sh --gpus 1 data/dev data/lang exp/ctc/version_0/checkpoints/epoch=26-vald_loss=0.0000.ckpt/ exp/ctc/version_0/decode_dev


. ./path.sh
. ./cmd.sh

gpus=1
batch_size=10

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: aling_data.sh [options] <data_dir> <lang_dir> <model_dir> <align_dir>"
  echo "     --batch_size                # default: 10, batch_size for aligning."
  echo "     --gpus                      # default: 4, the number of gpus used for aligning."
  echo "e.g.:"
  echo " $0 --gpus 4 --batch_size 10 data/train data/lang path/to/model path/for/output"
  exit 1
fi

data_dir=$1 
lang_dir=$2
model_dir=$3
align_dir=$4

mkdir -p $align_dir

align.py --data_dir $data_dir --lang_dir $lang_dir --model_dir $model_dir --output_dir $align_dir --jid 1 --gpus $gpus --batch_size $batch_size > $align_dir/align.log

