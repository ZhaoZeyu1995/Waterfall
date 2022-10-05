#!/bin/bash

. ./path.sh
. ./cmd.sh

lang=$1

num_token=$(cat $lang/k2/tokens.txt | wc -l)
#echo $num_token

func/generate_fake_topo.py $num_token > $lang/FakeTopo

gmm-init-mono $lang/FakeTopo 5 $lang/FakeTransitionModel $lang/FakeTree
