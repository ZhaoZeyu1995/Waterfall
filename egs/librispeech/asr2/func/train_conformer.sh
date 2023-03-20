#!/bin/bash

# Train a conformer model with a given configuration 
# e.g train_conformer.sh data=train960 model=conformer.sub4 training=convention specaug=convention

. ./path.sh || exit 1
. ./env.sh || exit 1
. ./cmd.sh || exit 1

set -e
set -u
set -o pipefail

if [[ $@ == '--help' ]]; then
  echo "Usage: train_conformer.sh <options>"
  echo "     --options                  # default: Null, the options that can be recognised by hydra."
  echo "e.g.:"
  echo " $0 data=train960 model=conformer.sub4 training=convention specaug=convention"
  exit 0
fi

opts="$@ hydra/job_logging=none hydra/hydra_logging=none"

echo "./train_conformer.py $opts"

./train_conformer.py $opts
#./train_conformer.py data.lang_dir=data/lang_bpe_5000_ctc training.gpus=7 --cfg job
