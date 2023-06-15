#!/bin/bash

# Train a transformer  model with a given configuration 
# e.g train_transformer.sh data=train960 model=transformer.sub4 training=convention specaug=convention

. ./path.sh || exit 1
. ./env.sh || exit 1
. ./cmd.sh || exit 1

set -e
set -u
set -o pipefail

if [[ $@ == '--help' ]]; then
  echo "Usage: train_transformer.sh <options>"
  echo "     --options                  # default: Null, the options that can be recognised by hydra."
  echo "e.g.:"
  echo " $0 data=train960 model=transformer.sub4 training=convention specaug=convention"
  exit 0
fi

opts="$@ hydra/job_logging=none hydra/hydra_logging=none"

echo "train_transformer.py $opts"

train_transformer.py $opts

