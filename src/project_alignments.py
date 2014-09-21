#!/usr/bin/env python3

"""
./project_alignments.py --alignments_file ../data/sw-en/data.aligned/train.sw-en.aligned.gdfa --corpus_file ../data/sw-en/data.tokenized/train.sw-en.filtered --hi_prob_dict ../data/sw-en/data.aligned/sw_en_dict_-0.7 --output ../work/sw_with_hi_prob_en --projection_counts ../work/sw_with_hi_prob_en.counts
"""
import argparse
import collections
import functools
import json
import itertools
import os

parser = argparse.ArgumentParser()
parser.add_argument("--alignments_file")
parser.add_argument("--corpus_file")
parser.add_argument("--hi_prob_dict")
parser.add_argument("--output")
parser.add_argument("--projection_counts")
args = parser.parse_args()

def LoadDict(filename):
  result = set()
  for line in open(filename):
    en, sw_words = line.strip().split(" ||| ")
    for sw in sw_words.split(" "):
      result.add( (sw, en) )
  return result

def AlignmentsToDict(line):
  result = collections.defaultdict(list)
  for pair in line.split():
    sw_i, en_i = pair.split("-")
    result[int(sw_i)].append(int(en_i))
  return result

def ExtractAlignments(sw_words, en_words, alignments_dict, hi_prob_pairs, projection_counter, sw_counter):
  result = []
  for sw_i, sw in enumerate(sw_words):
    sw_counter[sw] += 1
    en_translations = []
    for en_i in alignments_dict.get(sw_i, []):
      en = en_words[en_i]
      if (sw, en) in hi_prob_pairs:
        en_translations.append(en)
        projection_counter[(sw, en)] += 1
    out_sw = sw    
    if en_translations:
      out_sw += "_" + "_".join(en_translations)
    result.append(out_sw)
  return result

def main():
  hi_prob_pairs = LoadDict(args.hi_prob_dict)
  out_f = open(args.output, "w")
  projection_counter = collections.Counter()
  sw_counter = collections.Counter()
  for corpus_line, alignments_line in zip(open(args.corpus_file), open(args.alignments_file)):
    alignments_dict = AlignmentsToDict(alignments_line)
    sw_line, en_line = corpus_line.split(" ||| ")
    sw_to_en = ExtractAlignments(sw_line.split(), en_line.split(), alignments_dict, hi_prob_pairs, projection_counter, sw_counter)
    out_f.write(" ".join(sw_to_en))
    out_f.write("\n")
  if args.projection_counts:
    counts_f = open(args.projection_counts, "w")
    for (sw, en), count in projection_counter.items():
      counts_f.write("{} ||| {} ||| {} ||| {}\n".format(sw, en, count, sw_counter[sw]))

if __name__ == '__main__':
  main()
