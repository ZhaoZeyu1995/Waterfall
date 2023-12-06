#!/bin/bash 

. ./path.sh
. ./cmd.sh

# Decoding with FST-based decoder
# e.g. decode.sh test_dev93 data/lang exp/ctc/version_0


nj=10
max_active="6000"
acoustic_scale="6"
beams='16'

. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: decode.sh [options] <decode_set> <langdir> <exp_dir>"
  echo "     --nj                        # default: 10, the number of jobs."
  echo "     --max_active                # default: 10000, the max_active value for decode-faster in kaldi."
  echo "     --acoustic_scale            # default: 10.0, the acoustic_scale value for decode-faster in kaldi."
  echo "     --beams                     # default: 16, the decoding beam for decode-faster in kaldi."
  echo "e.g.:"
  echo " $0 test_dev93 data/lang_tg exp/ctc/version_0"
  exit 1
fi

decode_set=$1
langdir=$2
exp_dir=$3

graph=${langdir}/TLG.fst
output_dir="$exp_dir/decode_${decode_set}_hard"

#run.pl JOB=1:$nj $output_dir/split$nj/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $output_dir/output.1.scp $output_dir/split$nj/output.JOB.scp

for beam in $beams; do 
    for maxac in $max_active; do 
        for acwt in $acoustic_scale; do
            decode_dir=$output_dir/acwt_${acwt}-maxac_${maxac}-beam_${beam}
            mkdir -p decode_dir
            run.pl JOB=1:$nj $decode_dir/split${nj}/log/decode-faster.JOB.log decode-faster --max-active=$maxac --acoustic-scale=$acwt --beam=$beam --word-symbol-table=${langdir}/words.txt ${graph} "ark:copy-feats scp:$output_dir/split$nj/output.JOB.scp  ark:-|" "ark,t:$decode_dir/split$nj/hyp.wrd.JOB"

            for i in $(seq $nj); do
                utils/int2sym.pl -f 2- $langdir/words.txt $decode_dir/split$nj/hyp.wrd.$i | cat || exit 1
            done > $decode_dir/hyp.wrd.txt || exit 1

            post_process_decode.py $decode_dir/hyp.wrd.txt data/$decode_set/utt2spk > $decode_dir/hyp.wrd.trn

            sclite -r $output_dir/ref.wrd.trn.1 trn -h $decode_dir/hyp.wrd.trn trn -i rm -o all stdout > $decode_dir/results.wrd.txt
        done
    done
done

