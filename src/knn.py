#!/usr/bin/env python3

# Implements "Efficient k-nearest neighbor graph construction for generic similarity measures" in 
# http://www.ambuehler.ethz.ch/CDstore/www2011/proceedings/p577.pdf
#
# This is a library, no "main" here.

import collections
import itertools
import random
import sys
from operator import itemgetter

inf = float("inf")

class SortedArray(object):
  def __init__(self, k=sys.maxsize):
    self.array = []
    self.max = inf
    self.k = k

  def add(self, value, weight):
    if len(self.array) < self.k:
      if (value, weight) in self.array:
        return 0
      self.array.append((value, weight))
    else:
      if weight >= self.max:
        return 0
      if (value, weight) in self.array:
        return 0
      self.array[-1] = (value, weight)
    self.array = sorted(self.array, key=itemgetter(1))
    self.max = self.array[-1][1]
    return 1

  def __iter__(self):
    return iter(self.array)

class KNN(object):
  def __init__(self, vertices, k):
    """Vertices have to have a "Distance(other)" method."""
    self.vertices = vertices
    self.k = k
    self.Bmatrix = self.RandomSample()

  def RandomSample(self):
    result = {}
    for v in self.vertices:
      array = SortedArray(self.k)
      for u in random.sample(self.vertices, self.k):
        if u is not v:
          array.add(u, v.Distance(u))
      result[v] = array
    return result

  def Reverse(self, Bmatrix):
    result = collections.defaultdict(list)
    for v, sorted_array in Bmatrix.items():
      for u, weight in sorted_array:
        result[u].append((v, weight))
    return result

  def Run(self):
    iter_num = 0
    while True:
      iter_num += 1
      reverse = self.Reverse(self.Bmatrix)
      num_updates = 0
      for v in self.vertices:
        seen_u2 = set([v])
        for u1, w1 in list(itertools.chain(self.Bmatrix[v], reverse.get(v, []))):  # Btag[v]
          for u2, w2 in list(itertools.chain(self.Bmatrix[u1], reverse.get(u1, []))): # Btag[u1]
            if u2 not in seen_u2:
              weight = v.Distance(u2)
              num_updates += self.Bmatrix[v].add(u2, weight)
              seen_u2.add(u2)
      print("Iteration:", iter_num, "Num updates:", num_updates)
      if num_updates == 0:
        break
    return self.Bmatrix

if __name__ == '__main__':
  print("This is a library, not runnable by itself")
