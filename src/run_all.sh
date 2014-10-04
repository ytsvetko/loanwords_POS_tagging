#!/bin/bash

CORPUS_FILE=../data/sw-en/data.tokenized/train.sw-en.filtered

EN_POS_TAGGED=../data/sw-en/data.pos/all.en.pos

NUM_PROPAGATION_ITERATIONS=10

# china   {"NN": 0.008264462809917356, "NNP": 0.9917355371900827}
EN_POS_AGGREGATIONS=../data/sw-en/data.pos/all.en.pos.aggregated

ALIGNMENT_FILE_PREFIX=../data/sw-en/data.aligned/train.sw-en.aligned

#images taken by {"center_word": {"taken": 1.1111111111111112}, "cosine_denom_sum": 9.2134522572627, "count": 1.0, "left_context": {"the images": 1.4082482904638631}, "left_context_plus_right_word": {"the images by": 1.0}, "left_word_plus_right_context": {"images by afp": 1.0}, "name": ["images", "taken", "by"], "other_features": {"trigram": 0.9978582606932408}, "right_context": {"by afp": 1.0}, "trigram_context": {"the afp": 1.0}, "trigram_minus_center": {"images by": 1.0}}
VERTICES=../data/sw_vertices

#zone and do     pages and twitter 0.7426359567137443    television and radio 0.7426359567137443 invaders and their 0.7620413258437112   years and got 0.7620413258437112        government and its 0.7644555238560844   lake and potential 0.7656586776659642   courage and says 0.7687317442597076     congo and they 0.7692485457144588       read and / 0.783689046243066    planning and development 0.8245584062851776
KNN_GRAPH=../data/sw_knn_graph

# Log(0.36)~=-1.0
ALIGNMENT_PROB_THRESHOLD=-3.0

# academic ||| kitaalamu kitaaluma unakaribia wasomi wanataaluma
HI_PROB_DICT=../data/sw-en/data.aligned/sw_en_dict_${ALIGNMENT_PROB_THRESHOLD}

SW_WITH_HI_PROB_EN=../work/sw_with_hi_prob_en
SW_WITH_POS=../work/sw_with_pos

./aggregate_pos.py --pos_tagged ${EN_POS_TAGGED} --output ${EN_POS_AGGREGATIONS}

./dictionary_from_alignments.py --fwd_threshold ${ALIGNMENT_PROB_THRESHOLD} --rev_threshold ${ALIGNMENT_PROB_THRESHOLD} --fwd_prob ${ALIGNMENT_FILE_PREFIX}.fwd.probs --rev_prob ${ALIGNMENT_FILE_PREFIX}.rev.probs --out_file ${HI_PROB_DICT}

./project_alignments.py --en_pos_tags ${EN_POS_AGGREGATIONS} --alignments_file ${ALIGNMENT_FILE_PREFIX}.gdfa \
    --corpus_file ${CORPUS_FILE} --hi_prob_dict ${HI_PROB_DICT} --output ${SW_WITH_HI_PROB_EN} \
    --projection_counts ${SW_WITH_HI_PROB_EN}.counts

./propagate_pos.py --vertices_file ${VERTICES} --knn_graph_file ${KNN_GRAPH} --projections ${SW_WITH_HI_PROB_EN} \
    --num_iterations ${NUM_PROPAGATION_ITERATIONS} --output ${SW_WITH_POS}
