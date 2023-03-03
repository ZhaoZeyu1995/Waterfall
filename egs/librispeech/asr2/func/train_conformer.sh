#!/bin/bash

# Train a conformer model with the given configuration file
# e.g train_conformer.sh --train_config conf/train.yaml --expname ctc --gpus 4 data/train data/dev data/lang

. ./path.sh || exit 1
. ./env.sh || exit 1
. ./cmd.sh || exit 1


train_config="conf/train.conformer.k2.yaml"
expname=ctc
gpus=4
checkpoint=
load_weights_only=
batch_size=0
accumulate_grad_batches=1
spec_aug=true

. utils/parse_options.sh || exit 1

if [ $# != 3 ]; then
  echo "Usage: train_conformer.sh [options] <train_set> <dev_set> <lang_dir>"
  echo "     --train_config              # default: conf/train.yaml, the training configuration file."
  echo "     --expname                   # default: ctc, the output directory name in exp."
  echo "     --gpus                      # default: 4, the number of gpus used for training."
  echo "e.g.:"
  echo " $0 --train_config conf/train.yaml --gpus 4 --expname ctc data/train data/dev data/lang"
  exit 1
fi

train_set=$1
dev_set=$2
lang_dir=$3


if [ -z $checkpoint ]; then
    train_conformer.py --train_set $train_set --dev_set $dev_set --lang_dir $lang_dir --config $train_config --name $expname --gpus $gpus --batch_size $batch_size --accumulate_grad_batches $accumulate_grad_batches --spec_aug $spec_aug
else
    if [ $load_weights_only ]; then
        train_conformer.py --train_set $train_set --dev_set $dev_set --lang_dir $lang_dir --config $train_config --name $expname --gpus $gpus --checkpoint $checkpoint --load_weights_only true --batch_size $batch_size --accumulate_grad_batches $accumulate_grad_batches --spec_aug $spec_aug
    else
        train_conformer.py --train_set $train_set --dev_set $dev_set --lang_dir $lang_dir --config $train_config --name $expname --gpus $gpus --checkpoint $checkpoint --batch_size $batch_size --accumulate_grad_batches $accumulate_grad_batches --spec_aug $spec_aug
    fi
fi
