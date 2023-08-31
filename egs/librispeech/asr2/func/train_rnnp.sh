#!/bin/bash

# Train an RNNP model with the given configuration file
# e.g train_rnnp.sh data=yesno model=rnnp training=convention

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
  echo " $0 data=yesno model=rnnp training=convention"
  exit 0
fi

opts="$@ hydra/job_logging=none hydra/hydra_logging=none"

echo "train_rnnp.py $opts"

train_rnnp.py $opts
