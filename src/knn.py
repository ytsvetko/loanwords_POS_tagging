#!/usr/bin/env python3

# Implements "Efficient k-nearest neighbor graph construction for generic similarity measures" in 
# http://www.ambuehler.ethz.ch/CDstore/www2011/proceedings/p577.pdf
#
# This is a library, no "main" here.

import collections
import itertools
import time
import random
import sys
from operator import itemgetter

inf = float("inf")

class SortedArray(object):
  def __init__(self, k=sys.maxsize):
    self.array = []
    self.max = inf
    self.k = k

  def add(self, value, distance):
    if len(self.array) < self.k:
      if (value, distance) in self.array:
        return 0
      self.array.append((value, distance))
    else:
      if distance >= self.max:
        return 0
      if (value, distance) in self.array:
        return 0
      self.array[-1] = (value, distance)
    self.array = sorted(self.array, key=itemgetter(1))
    self.max = self.array[-1][1]
    return 1

  def __iter__(self):
    return iter(self.array)

class KNN(object):
  def __init__(self, vertices, k, filename=None):
    """Vertices have to have a "Distance(other)" method."""
    self.vertices = vertices
    self.vertices_list = list(self.vertices.values())
    self.k = k
    if filename:
      self.Bmatrix = self.LoadMatrix(filename)
    else:
      self.Bmatrix = self.RandomSample()

  def RandomSample(self):
    result = {}
    for v in self.vertices_list:
      array = SortedArray(self.k)
      for u in random.sample(self.vertices_list, self.k):
        if u is not v:
          array.add(u, v.Distance(u))
      result[v] = array
    return result

  def Reverse(self, Bmatrix):
    def KSortedArray():
      return SortedArray(self.k)
    result = collections.defaultdict(KSortedArray)
    for v, sorted_array in Bmatrix.items():
      for u, distance in sorted_array:
        result[u].add(v, distance)
    return result

  def Run(self, save_filename=None):
    iter_num = 0
    while True:
      iter_num += 1
      print(time.strftime("%Y/%m/%d %H:%M:%S"), "Building reverse matrix")
      reverse = self.Reverse(self.Bmatrix)
      num_updates = 0
      for index, v in enumerate(self.vertices_list):
        if index % 10000 == 0:
          print(time.strftime("%Y/%m/%d %H:%M:%S"), index)
        seen_u2 = set([v])
        for u1, w1 in list(itertools.chain(self.Bmatrix[v], reverse.get(v, []))):  # Btag[v]
          for u2, w2 in list(itertools.chain(self.Bmatrix[u1], reverse.get(u1, []))): # Btag[u1]
            if u2 not in seen_u2:
              distance = v.Distance(u2)
              num_updates += self.Bmatrix[v].add(u2, distance)
              seen_u2.add(u2)
      print(time.strftime("%Y/%m/%d %H:%M:%S"), "Iteration:", iter_num, "Num updates:", num_updates)
      if save_filename:
        self.SaveMatrix(save_filename)
      if num_updates == 0:
        break
    return self.Bmatrix

  def GetMatrix(self, distance_threshold=inf):
    result = {}
    for v, array in self.Bmatrix.items():
      result_array = []
      for nn, distance in array:
        if distance > distance_threshold:
          break
        result_array.append((nn, distance))
      result[v] = result_array
    return result

  def SaveMatrix(self, filename):
    with open(filename, "w") as f:
      for v, knn_array in sorted(self.Bmatrix.items()):
        knn_str = "\t".join([" ".join(u.name) + " " + str(distance) for (u, distance) in knn_array])
        f.write("{}\t{}\n".format(" ".join(v.name), knn_str))

  def LoadMatrix(self, filename):
    def ParseToken(token):
      name_distance = token.split(" ")
      return tuple(name_distance[:-1]), float(name_distance[-1])
    matrix = {}
    for line in open(filename):
      tokens = line.strip().split("\t")
      v_name = tuple(tokens[0].split(" "))
      v = self.vertices[v_name]
      array = SortedArray(self.k)
      for token in tokens[1:]:
        name, distance = ParseToken(token)
        array.add(self.vertices[name], distance)
      matrix[v] = array
    return matrix

if __name__ == '__main__':
  print("This is a library, not runnable by itself")
