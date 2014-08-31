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
import itertools
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="../data/sw-en/data.tokenized/train.sw-en.filtered")
parser.add_argument("--k", default=5, type=int)
args = parser.parse_args()

class Vertex(object):
  def __init__(self):
    self.trigram_context = collections.defaultdict(float)
    self.trigram = 0.0
    self.left_context = collections.defaultdict(float)
    self.right_context = collections.defaultdict(float)
    self.center_word = collections.defaultdict(float)
    self.trigram_minus_center = collections.defaultdict(float)
    self.left_word_plus_right_context = collections.defaultdict(float)
    self.left_context_plus_right_word = collections.defaultdict(float)
    #TODO self.has_suffix = False
    self.cosine_denom_sum = 0.0
    self.knn = [(-2.0, None)]*args.k # tuple array (distance, vertex)

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

  def UpdateKNN(self, vertex):
    cosine = self.Cosine(vertex)
    if cosine > self.knn[-1][0]:
      self.knn[-1] = (cosine, vertex)
      self.knn = sorted(self.knn, reverse=True, key=itemgetter(0))
    
  def __repr__(self):
   return "*******************Vertex: {}*******************\n  Trigram_context: {}\n  Trigram: {}".format(self.name, self.trigram_context, self.trigram)

def LineToNgrams(line, n):
  # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/ 
  line = ["PAD_START"] + line.split() + ["PAD_END"]
  return zip(*[line[i:] for i in range(n)])

def main():
  vertices = collections.defaultdict(Vertex) # key -trigram tuple
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

  print("Calculating KNN...")
  for (vertex1, vertex2) in itertools.product(vertices.values(),vertices.values()):
    if vertex1 is vertex2:
      continue
    vertex1.UpdateKNN(vertex2)
      
    
if __name__ == '__main__':
  main()
