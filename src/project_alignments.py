#!/usr/bin/env python3

"""
./project_alignments.py --alignments_file ../data/sw-en/data.aligned/train.sw-en.aligned.gdfa --corpus_file ../data/sw-en/data.tokenized/train.sw-en.filtered --hi_prob_dict ../data/sw-en/data.aligned/sw_en_dict_-0.7 --output ../work/sw_with_hi_prob_en --projection_counts ../work/sw_with_hi_prob_en.counts
"""
import argparse
import collections
import operator
import json
import itertools
import os

parser = argparse.ArgumentParser()
parser.add_argument("--en_pos_tags")
parser.add_argument("--min_sw_word_count", type=int, default=2)
parser.add_argument("--alignments_file")
parser.add_argument("--corpus_file")
parser.add_argument("--hi_prob_dict")
parser.add_argument("--output")
parser.add_argument("--projection_counts")
args = parser.parse_args()

def LoadDict(filename, en_pos_tags):
  result = set()
  for line in open(filename):
    en, sw_words = line.strip().split(" ||| ")
    if en in en_pos_tags:
      for sw in sw_words.split(" "):
        result.add( (sw, en) )
    else:
      print("No POS tags for:", en)
  return result

def AlignmentsToDict(line):
  result = collections.defaultdict(list)
  for pair in line.split():
    sw_i, en_i = pair.split("-")
    result[int(sw_i)].append(int(en_i))
  return result

def CountProjections(sw_words, en_words, alignments_dict, hi_prob_pairs, projection_counter, sw_counter):
  for sw_i, sw in enumerate(sw_words):
    sw_counter[sw] += 1
    en_translations = set()
    for en_i in alignments_dict.get(sw_i, []):
      en = en_words[en_i]
      if (sw, en) in hi_prob_pairs:
        projection_counter[sw][en] += 1

def GetProjections(sw, en, projection_counter, en_pos_tags):
  def MulScalarByVector(scalar, vector_dict):
    return {k:v*scalar for k,v in vector_dict.items()}
  def AddVector(v1, v2):
    result = {}
    for k in set(v1.keys()) | set(v2.keys()):
      result[k] = v1.get(k, 0.0) + v2.get(k, 0.0)
    return result
  def DivideVector(v1, v2):
    result = {}
    for k, v in v1.items():
      result[k] = v/v2[k]
    return result
  all_en_counters = projection_counter[sw]
  nominator = MulScalarByVector(all_en_counters[en], en_pos_tags[en])
  denom = {}
  for other_en, count in all_en_counters.items():
    denom = AddVector(denom, MulScalarByVector(count, en_pos_tags[other_en]))
  return DivideVector(nominator, denom)

def ExtractAlignments(sw_words, en_words, alignments_dict, hi_prob_pairs,
                      projection_counter, en_pos_tags, sw_counts):
  result = []
  for sw_i, sw in enumerate(sw_words):
    num_en_translations = 0
    if sw_counts.get(sw, 0) > args.min_sw_word_count:
      en_translation = None
      for en_i in alignments_dict.get(sw_i, []):
        en = en_words[en_i]
        if (sw, en) in hi_prob_pairs:
          if en_translation == en:
            continue
          num_en_translations += 1
          en_translation = en
          projection = GetProjections(sw, en, projection_counter, en_pos_tags)
    if num_en_translations == 1:
      result.append( (sw, en_translation, projection) )
    else:
      result.append( (sw, None, None) )
  return result

def LoadEnPosTags(filename):
  # china   {"NN": 0.008264462809917356, "NNP": 0.9917355371900827}
  result = {}
  for line in open(filename):
    word, json_str = line.strip().split("\t")
    result[word] = json.loads(json_str)
  return result

def main():
  en_pos_tags = LoadEnPosTags(args.en_pos_tags)
  hi_prob_pairs = LoadDict(args.hi_prob_dict, en_pos_tags)
  out_f = open(args.output, "w")

  # First pass: Count projections
  projection_counter = collections.defaultdict(collections.Counter)
  sw_counter = collections.Counter()
  for corpus_line, alignments_line in zip(open(args.corpus_file), open(args.alignments_file)):
    alignments_dict = AlignmentsToDict(alignments_line)
    sw_line, en_line = corpus_line.split(" ||| ")
    CountProjections(sw_line.split(), en_line.split(), alignments_dict,
                     hi_prob_pairs, projection_counter, sw_counter)
  if args.projection_counts:
    counts_f = open(args.projection_counts, "w")
    for sw, en_counter in sorted(projection_counter.items(), key=operator.itemgetter(0)):
      for en, count in sorted(en_counter.items()):
        counts_f.write("{} ||| {} ||| {} ||| {}\n".format(sw, en, count, sw_counter[sw]))

  # Second pass: annotate SW words with POS from hi-prob alignments.
  for corpus_line, alignments_line in zip(open(args.corpus_file), open(args.alignments_file)):
    alignments_dict = AlignmentsToDict(alignments_line)
    sw_line, en_line = corpus_line.split(" ||| ")
    sw_to_en = ExtractAlignments(sw_line.split(), en_line.split(), alignments_dict,
                                 hi_prob_pairs, projection_counter, en_pos_tags, sw_counter)
    tags = []
    for sw, en, pos in sw_to_en:
      out_f.write(sw)
      if en is None:
        tags.append("")
      else:
        tags.append((en + " " + json.dumps(pos, sort_keys=True)))
      out_f.write(" ")
    out_f.write(" ||| ")
    out_f.write("\t".join(tags))
    out_f.write("\n")

if __name__ == '__main__':
  main()
