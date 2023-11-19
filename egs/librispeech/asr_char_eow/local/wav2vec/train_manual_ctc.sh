#!/bin/bash

# Training the wav2vec 2 model based on my implementation of CTC
# e.g train.sh --config conf/train.yaml --name manual_ctc data/train data/dev data/lang


. ./path.sh
. ./cmd.sh

config=conf/train.yaml
name=manual_ctc
gpus=4

. ./utils/parse_options.sh

train_set=$1
dev_set=$2
lang_dir=$3

train_manual_ctc.py --train_set $train_set --dev_set $dev_set --lang_dir $lang_dir --config $config --name $name --gpus $gpus
