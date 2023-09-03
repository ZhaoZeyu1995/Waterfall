#!/usr/bin/env bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus
# http://www.openslr.org/resources (Mirror).
#
# Note: this only trains on the tedlium-1 data, there is now a second
# release which we plan to incorporate in a separate directory, e.g
# s5b or s5-release2.
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Johs Hopkins University (Author: Daniel Povey)
# Apache 2.0
#

. ./cmd.sh
. ./path.sh

nj=40
decode_nj=8

stage=0

. utils/parse_options.sh  # accept options.. you can run this run.sh with the
                          # --stage option, for instance, if you don't want to
                          # change it in the script.

# Data preparation
if [ $stage -le 0 ]; then
  local/download_data.sh || exit 1

  local/prepare_data.sh || exit 1

  local/prepare_dict.sh || exit 1

  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp || exit 1

  local/prepare_lm.sh || exit 1

fi
