#!/usr/bin/env python

"""
./dictionary_from_alignments.py --fwd_threshold -1.0 --rev_threshold -1.0 --fwd_prob ../data/sw-en/data.aligned/train.sw-en.aligned.fwd.probs --rev_prob ../data/sw-en/data.aligned/train.sw-en.aligned.rev.probs --out_file ../data/sw-en/data.aligned/sw_en_dict_-1.0
"""

import argparse
import sys
import collections
import codecs
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--fwd_probs", required=True)
parser.add_argument("--rev_probs", required=True)
parser.add_argument("--fwd_threshold", default=-5.0, type=float)
parser.add_argument("--rev_threshold", default=-5.0, type=float)
parser.add_argument("--out_filename", required=True)
args = parser.parse_args()

def GetProbableTranslations(in_file, en_is_first_column, threshold):
  # returns dict( en -> set(fr) )
  result = collections.defaultdict(set)
  for line in open(in_file):
    try:
      line = line.decode("utf-8")
    except Exception as e:
      print e
      continue
    try:
      en, fr, log_prob = line.split()
    except:
      continue

    if float(log_prob) < threshold:
      continue
    if not en_is_first_column:
      en, fr = fr, en
    result[en].add(fr)

  return result


def main():
  fwd_dict = GetProbableTranslations(args.fwd_probs, False, args.fwd_threshold)
  rev_dict = GetProbableTranslations(args.rev_probs, True, args.rev_threshold)
  overlapping_keys = set(fwd_dict.keys()) & set(rev_dict.keys())
  out_file = codecs.open(args.out_filename, "w", "utf-8")
  for k in sorted(overlapping_keys):
    if k.isalpha():
      overlapping_words = fwd_dict[k] & rev_dict[k]
      overlapping_words -= set([k])
      if len(overlapping_words) > 0:
        out_file.write(u"{} ||| {}\n".format(k, u" ".join(overlapping_words)))

if __name__ == '__main__':
    main()

