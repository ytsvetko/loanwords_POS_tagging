#!/bin/bash

CORPUS_FILE=../data/sw-en/data.tokenized/train.sw-en.filtered

EN_POS_TAGGED=../data/sw-en/data.pos/all.en.pos
EN_POS_AGGREGATIONS=../data/sw-en/data.pos/all.en.pos.aggregated

ALIGNMENT_FILE_PREFIX=../data/sw-en/data.aligned/train.sw-en.aligned
ALIGNMENT_PROB_THRESHOLD=-0.7
HI_PROB_DICT=../data/sw-en/data.aligned/sw_en_dict_${ALIGNMENT_PROB_THRESHOLD}

SW_WITH_HI_PROB_EN=../work/sw_with_hi_prob_en

./aggregate_pos.py --pos_tagged ${EN_POS_TAGGED} --output ${EN_POS_AGGREGATIONS}

./dictionary_from_alignments.py --fwd_threshold ${ALIGNMENT_PROB_THRESHOLD} --rev_threshold ${ALIGNMENT_PROB_THRESHOLD} --fwd_prob ${ALIGNMENT_FILE_PREFIX}.fwd.probs --rev_prob ${ALIGNMENT_FILE_PREFIX}.rev.probs --out_file ${HI_PROB_DICT}

./project_alignments.py --alignments_file ${ALIGNMENT_FILE_PREFIX}.gdfa --corpus_file ${CORPUS_FILE} --hi_prob_dict ${HI_PROB_DICT} --output ${SW_WITH_HI_PROB_EN} --projection_counts ${SW_WITH_HI_PROB_EN}.counts
