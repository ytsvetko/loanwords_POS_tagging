#!/usr/bin/env python3

"""
./aggregate_pos.py --pos_tagged <filename> --output <filename>
"""
import argparse
import collections
import functools
import json
import itertools
import os

POS_CONVERSIONS = {
    "NNS": "NN",
#    "VBD": "VB",
#    "VBN": "VB",
}

parser = argparse.ArgumentParser()
parser.add_argument("--pos_tagged")
parser.add_argument("--output")
args = parser.parse_args()

def CountPosTags(infile):
  result = collections.defaultdict(collections.Counter)
  for line in infile:
    for token in line.split():
      word, pos_tag = token.rsplit("_", 1)
      word = word.lower()
      pos_tag = POS_CONVERSIONS.get(pos_tag, pos_tag)
      result[word][pos_tag] += 1
  return result

def AggregatePosTags(all_counts):
  result = {}
  for word, counts in all_counts.items():
    total = sum(counts.values())
    result[word] = {pos_tag: count/total for pos_tag, count in counts.items()}
  return result

def main():
  all_aggregations = AggregatePosTags(CountPosTags(open(args.pos_tagged)))
  out_f = open(args.output, "w")
  for word, aggregations  in all_aggregations.items():
    out_f.write("{}\t{}\n".format(word, json.dumps(aggregations, sort_keys=True)))

if __name__ == '__main__':
  main()
