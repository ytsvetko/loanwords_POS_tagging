#!/usr/bin/env python3

# Implements "Foreign language graph" in 
# http://www.aclweb.org/anthology/P11-1061.pdf
# Unsupervised part-of-speech tagging with bilingual graph-based projections

"""
./graph_f.py --corpus ../data/sw-en/data.tokenized/mono.sw
"""
import argparse
import collections
import math
import functools
import json
import itertools
import os
import knn
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="../data/sw-en/data.tokenized/train.sw-en.filtered")
parser.add_argument("--vertices_file", default="../data/sw_vertices")
parser.add_argument("--graph_file", default="../data/sw_knn_graph")
parser.add_argument("--k", default=5, type=int, help="k in KNN")
parser.add_argument("-f", action="store_true", help="Force re-computation of KNN graph")
args = parser.parse_args()

class Vertex(object):
  def __init__(self, s=None):
    if s is not None:
      self.loads(s)     
    else:
      self.name = None
      self.trigram_pmi = None
      self.trigram = 0.0
      self.cosine_denom_sum = 0.0
      self.trigram_context = collections.defaultdict(float)
      self.left_context = collections.defaultdict(float)
      self.right_context = collections.defaultdict(float)
      self.center_word = collections.defaultdict(float)
      self.trigram_minus_center = collections.defaultdict(float)
      self.left_word_plus_right_context = collections.defaultdict(float)
      self.left_context_plus_right_word = collections.defaultdict(float)
      #TODO self.has_suffix = False

  def GetDicts(self):
    return[self.trigram_context, self.left_context, self.right_context, 
        self.center_word, self.trigram_minus_center, self.left_word_plus_right_context, 
        self.left_context_plus_right_word]

  def Update(self, five_gram):
    """Updates the vertex given a 5-gram"""
    self.name = five_gram[1:-1]
    self.trigram_context[(five_gram[0], five_gram[-1])] += 1
    self.trigram += 1
    self.left_context[five_gram[:2]] += 1
    self.right_context[five_gram[-2:]] += 1
    self.center_word[(five_gram[2],)] += 1
    self.trigram_minus_center[(five_gram[1], five_gram[3])] += 1
    self.left_word_plus_right_context[(five_gram[1], five_gram[3], five_gram[4])] += 1
    self.left_context_plus_right_word[(five_gram[0], five_gram[1], five_gram[3])] += 1
    #TODO self.has_suffix

  def PMI(self, d1, counts_d1, d2, counts_d2):
    d = {}
    for k, v in d1.items():
      d[k] = math.log(v/counts_d1) - math.log(d2[k]/counts_d2)
    return d

  def UpdatePMI(self, corpus): 
    self.trigram_context = self.PMI(self.trigram_context, self.trigram, 
        corpus.trigram_context, corpus.trigram) # only context, without tri-gram
    self.left_context = self.PMI(self.left_context, self.trigram, 
        corpus.left_context, corpus.trigram)
    self.right_context = self.PMI(self.right_context, self.trigram, 
        corpus.right_context, corpus.trigram)
    self.center_word = self.PMI(self.center_word, self.trigram, 
        corpus.center_word, corpus.trigram)
    self.trigram_minus_center = self.PMI(self.trigram_minus_center, self.trigram, 
        corpus.trigram_minus_center, corpus.trigram)
    self.left_word_plus_right_context = self.PMI(self.left_word_plus_right_context, self.trigram, 
        corpus.left_word_plus_right_context, corpus.trigram)
    self.left_context_plus_right_word = self.PMI(self.left_context_plus_right_word, self.trigram, 
        corpus.left_context_plus_right_word, corpus.trigram)
    self.trigram_pmi = math.log(corpus.trigram/self.trigram)
    self.UpdateCosineDenomSum()

  def UpdateCosineDenomSum(self):
    def GetSum(d):
      return sum([v**2 for v in d.values()])
    self.cosine_denom_sum = sum([GetSum(d) for d in self.GetDicts()])
    self.cosine_denom_sum += self.trigram_pmi

  def Cosine(self, vertex):
    def GetNumerator(d1, d2):
      intersection = set(d1.keys()) & set(d2.keys())
      return sum([d1[k]*d2[k] for k in intersection])
    numerator = sum([GetNumerator(d1, d2) for (d1, d2) in zip(self.GetDicts(), vertex.GetDicts())])
    numerator += self.trigram_pmi*vertex.trigram_pmi
    denominator = math.sqrt(self.cosine_denom_sum) * math.sqrt(vertex.cosine_denom_sum)
    if not denominator:
      return 0.0
    else:
      return numerator/denominator
    #TODO self.has_suffix = False

  @functools.lru_cache(maxsize=1000000)
  def Distance(self, other):
    return 1-self.Cosine(other)

  def dumps(self):
    def StrDict(d):
      result = {}
      for k, v in d.items():
        result[" ".join(k)] = v
      return result

    all_dicts = {}
    for k, v in vars(self).items():
      if isinstance(v, dict):
        all_dicts[k] = StrDict(v)
      else:
        all_dicts[k] = v
    return "{}\t{}\n".format(' '.join(self.name), json.dumps(all_dicts, sort_keys=True))

  def loads(self, s):
    def TupleDict(d):
      result = {}
      for k, v in d.items():
        result[tuple(k.split())] = v
      return result

    name_str, dict_str = s.strip().split('\t')
    for attr, val in json.loads(dict_str).items():
      if isinstance(val, dict):
        val = TupleDict(val)
      if isinstance(val, list):
        val = tuple(val)
      setattr(self, attr, val)

  def __repr__(self):
   return "Vertex: {}".format(self.name)

  def __lt__(self, other):
    if self.name < other.name:
      return True
    return False
    
def LineToNgrams(line, n):
  # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/ 
  line = ["PAD_START"] + line.split() + ["PAD_END"]
  return zip(*[line[i:] for i in range(n)])

def main():
  vertices = collections.defaultdict(Vertex) # key -trigram tuple
  if args.f or not os.path.exists(args.vertices_file):
    corpus = Vertex()
    print("Loading tri-grams...")
    for line in open(args.corpus):
      fivegrams = LineToNgrams(line, 5)
      for fivegram in fivegrams:
        vertices[fivegram[1:-1]].Update(fivegram)
        corpus.Update(fivegram)
    print("Number of Vertices: {}".format(len(vertices)))

    print("Updating PMI...")
    for trigram, vertex in vertices.items():
      vertex.UpdatePMI(corpus)

    print("Write vertices to file")
    f = open(args.vertices_file, "w")
    for v in vertices.values():
      f.write(v.dumps())
  else:
    print("Read vertices from file")
    for line in open(args.vertices_file):
      v = Vertex(line)
      vertices[v.name] = v

  print("Building KNN graph")
  knn_graph_builder = knn.KNN(list(vertices.values()), args.k)
  knn_matrix = knn_graph_builder.Run()
  if args.f or not os.path.exists(args.graph_file):
    f = open(args.graph_file, "w")
    for v, knn_array in sorted(knn_matrix.items()):
      knn_str = "\t".join([" ".join(u.name) + " " + str(distance) for (u, distance) in knn_array])
      f.write("{}\t{}\n".format(" ".join(v.name), knn_str))      
if __name__ == '__main__':
  main()
