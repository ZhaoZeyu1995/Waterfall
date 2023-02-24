#!/bin/bash

. ./path.sh
. ./env.sh
. ./cmd.sh

loglikeli_scp=$1
token_list=$2

# usage : calculate_argmax_in_output_logliklihood.sh loglikeli_scp token_list report_output

if [ -z $3 ]; then
    report_output="$(dirname $loglikeli_scp)/statistics_token.txt"
else
    report_output=$3
fi

echo $0

calculate_argmax_in_outputs.py $loglikeli_scp $token_list ${report_output}
