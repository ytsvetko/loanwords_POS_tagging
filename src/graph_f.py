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

if __name__ == '__main__':
  global parser, args
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", default="../data/sw-en/data.tokenized/train.sw-en.filtered.en")
  parser.add_argument("--vertices_file", default="../data/sw_vertices")
  parser.add_argument("--graph_file", default="../data/sw_knn_graph")
  parser.add_argument("--k", default=10, type=int, help="k in KNN")
  parser.add_argument("-f", action="store_true", help="Force re-computation of KNN graph")
  args = parser.parse_args()


class Vertex(object):
  def __init__(self, s=None):
    if s is not None:
      self.loads(s)     
    else:
      self.name = None
      self.count = 0.0
      self.cosine_denom_sum = 0.0
      self.trigram_context = collections.defaultdict(float)
      self.left_context = collections.defaultdict(float)
      self.right_context = collections.defaultdict(float)
      self.center_word = collections.defaultdict(float)
      self.trigram_minus_center = collections.defaultdict(float)
      self.left_word_plus_right_context = collections.defaultdict(float)
      self.left_context_plus_right_word = collections.defaultdict(float)
      self.other_features = collections.defaultdict(float)

  def GetDicts(self):
    return[self.trigram_context, self.left_context, self.right_context, 
        self.center_word, self.trigram_minus_center, self.left_word_plus_right_context, 
        self.left_context_plus_right_word, self.other_features]

  def Update(self, five_gram):
    """Updates the vertex given a 5-gram"""
    self.name = five_gram[1:-1]
    self.trigram_context[(five_gram[0], five_gram[-1])] += 1
    self.count += 1    
    self.left_context[five_gram[:2]] += 1
    self.right_context[five_gram[-2:]] += 1
    self.center_word[(five_gram[2],)] += 1
    self.trigram_minus_center[(five_gram[1], five_gram[3])] += 1
    self.left_word_plus_right_context[(five_gram[1], five_gram[3], five_gram[4])] += 1
    self.left_context_plus_right_word[(five_gram[0], five_gram[1], five_gram[3])] += 1
    self.other_features[("trigram",)] += 1
    #TODO
    # if self.HasSuffix():
    #   self.other_features[("has_suffix",)] += 1

  def ConvertToPMI(self, d1, counts_d1, d2, counts_d2):
    for k, v in d1.items():
      d1[k] = math.log(v/counts_d1) - math.log(d2[k]/counts_d2)
      #d1[k] = (v/counts_d1)/(d2[k]/counts_d2)

  def UpdatePMI(self, corpus): 
    for d, corpus_d in zip(self.GetDicts(), corpus.GetDicts()):
      self.ConvertToPMI(d, self.count, corpus_d, corpus.count)
    self.other_features[("trigram",)] = math.log(self.count / corpus.count)
    self.UpdateDenomSums()

  def UpdateDenomSums(self):
    def GetSum(d):
      return sum([v**2 for v in d.values()])
    self.cosine_denom_sum = sum([GetSum(d) for d in self.GetDicts()])
    self.sum_similarity_denom = sum([sum(d.values()) for d in self.GetDicts()])

  def Cosine(self, vertex):
    def GetNumerator(d1, d2):
      intersection = set(d1.keys()) & set(d2.keys())
      return sum([d1[k]*d2[k] for k in intersection])
    numerator = sum([GetNumerator(d1, d2) for (d1, d2) in zip(self.GetDicts(), vertex.GetDicts())])
    denominator = math.sqrt(self.cosine_denom_sum) * math.sqrt(vertex.cosine_denom_sum)
    if not denominator:
      return 0.0
    else:
      return numerator/denominator
      
  def Similarity(self, vertex):
    def GetSum(d1, d2):
      combined = set(d1.keys()) & set(d2.keys())
      return sum( (d1[k] + d2[k] for k in combined) )
    nominator = sum( (GetSum(d1, d2) for (d1, d2) in zip(self.GetDicts(), vertex.GetDicts())) )
    denominator = self.sum_similarity_denom + vertex.sum_similarity_denom
    return nominator/denominator

  ##@functools.lru_cache(maxsize=1000000)
  def Distance(self, other):
    #return 1-self.Cosine(other)
    return 1-self.Similarity(other)

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
    self.UpdateDenomSums()

  def __repr__(self):
   return "Vertex: {}".format(self.name)

  def __lt__(self, other):
    if self.name < other.name:
      return True
    return False

def Normalize(vertices, corpus):
  sum_dict = collections.defaultdict(float)
  for v in vertices.values():
    for i, d in enumerate(v.GetDicts()):
      for k, val in d.items():
        sum_dict[(i, k)] += val
  
  variance_dict = collections.defaultdict(float)
  for v in vertices.values():
    for i, (d, corpus_d) in enumerate(zip(v.GetDicts(), corpus.GetDicts())):
      for k, val in d.items():
        average = sum_dict[(i, k)] / corpus_d[k]
        variance_dict[(i, k)] += (val - average)**2
        d[k] -= average

  for v in vertices.values():
    for i, d in enumerate(v.GetDicts()):
      for k in d:
        sigma = variance_dict[(i, k)]**0.5
        if sigma != 0.0:
          d[k] /= sigma
        d[k] += 1
    v.UpdateDenomSums()


def LineToNgrams(line, n):
  # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/ 
  line = ["PAD_START", "PAD_START"] + line.split() + ["PAD_END", "PAD_END"]
  return zip(*[line[i:] for i in range(n)])

def DebugFindKNN(trigram, k, vertices, do_print=True):
  v = vertices.get(tuple(trigram.split()), None)
  if v is None:
    print(trigram, "not found")
    return
  array = knn.SortedArray(k)
  for u in vertices.values():
    if u is v:
      continue
    array.add(u, v.Distance(u))
  if do_print:
    print("\n".join([" ".join(u.name) + " " + str(distance) for (u, distance) in reversed(list(array))]))
  return array

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
    
    print("Normalizing features")
    Normalize(vertices, corpus)

    print("Write vertices to file")
    with open(args.vertices_file, "w") as f:
      for v in vertices.values():
        f.write(v.dumps())
  else:
    print("Read vertices from file")
    for line in open(args.vertices_file):
      v = Vertex(line)
      vertices[v.name] = v
    print("Number of Vertices: {}".format(len(vertices)))

  ###### DEBUG
  #DebugFindKNN('have to do', 10, vertices)
  #import pdb; pdb.set_trace()
  ###### DEBUG END

  if args.f or not os.path.exists(args.graph_file):
    print("Building KNN graph")
    knn_graph_builder = knn.KNN(vertices, args.k)
    knn_matrix = knn_graph_builder.Run(args.graph_file)
  else:
    print("Loading KNN graph")
    knn_graph_builder = knn.KNN(vertices, args.k, args.graph_file)

if __name__ == '__main__':
  main()
