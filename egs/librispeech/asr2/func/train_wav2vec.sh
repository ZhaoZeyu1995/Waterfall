#!/bin/bash

# Fine-tune a pre-trained wav2vec model with a given configuration 
# e.g train_wav2vec.sh data=train960 model=base960h training=convention

. ./path.sh || exit 1
. ./env.sh || exit 1
. ./cmd.sh || exit 1


set -e
set -u
set -o pipefail

if [[ $@ == '--help' ]]; then
  echo "Usage: train_wav2vec.sh <options>"
  echo "     --options                  # default: Null, the options that can be recognised by hydra."
  echo "e.g.:"
  echo " $0 data=train960 model=base960h training=convention"
  exit 0
fi

opts="$@ hydra/job_logging=none hydra/hydra_logging=none"

echo "train_wav2vec.py $opts"

train_wav2vec.py $opts
