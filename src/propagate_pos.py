#!/usr/bin/env python3

"""
./propagate_pos.py --vertices_file ../data/sw_vertices_sw --knn_graph_file ../data/sw_knn_graph_sw \
    --projections ../work/sw_with_hi_prob_en --num_iterations 10 --output ../work/sw_with_pos
"""
import argparse
import collections
import operator
import json
import itertools
import os
import graph_f
import knn
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--vertices_file")
parser.add_argument("--num_iterations", type=int, default=10)
parser.add_argument("--knn_distance_threshold", type=float, default=0.8)
parser.add_argument("--nu", type=float, default=2e-6)
parser.add_argument("--knn_graph_file")
parser.add_argument("--projections")
parser.add_argument("--output")
args = parser.parse_args()


def LoadProjections(filename, vertices):
  all_pos = set()
  def ParseProjection(s):
    if not s:
      return None
    en, json_str = s.split(" ", 1)
    pos_dict = json.loads(json_str)
    for pos in pos_dict.keys():
      all_pos.add(pos)
    return pos_dict

  projections = {}
  for line in open(filename):
    sw_line, line_projections = line[:-1].split(" ||| ")
    sw_line = ["PAD_START"] + sw_line.split() + ["PAD_END"]
    line_projections = "\t" + line_projections + "\t"
    line_projections = [ParseProjection(s) for s in line_projections.split("\t")]
    for i in range(1, len(sw_line)-2):
      if line_projections[i]:
        v_name = tuple(sw_line[i-1:i+2])
        if v_name in vertices:
          projections[vertices[v_name]] = line_projections[i]
  return projections, all_pos


def MulScalarByVector(scalar, vector_dict):
  return {k:v*scalar for k,v in vector_dict.items()}

def AddVector(v1, v2):
  result = {}
  for k in set(v1.keys()) | set(v2.keys()):
    result[k] = v1.get(k, 0.0) + v2.get(k, 0.0)
  return result

def main():
  vertices ={}
  print("Read vertices from file")
  for line in open(args.vertices_file):
    v = graph_f.Vertex(line)
    vertices[v.name] = v
  print("Number of Vertices: {}".format(len(vertices)))
  
  print("Loading KNN graph")
  knn_graph = knn.KNN(vertices, sys.maxsize, args.knn_graph_file).GetMatrix(args.knn_distance_threshold)

  print("Loading projections")
  initial_vertex_projections, all_pos = LoadProjections(args.projections, vertices)
  
  uniform_pos = {pos:1/len(all_pos) for pos in all_pos}

  current_projections = initial_vertex_projections
  for i in range(args.num_iterations):
    print("Iteration:", i+1)
    new_projections = {}
    for v in vertices.values():
      if v in initial_vertex_projections:
        new_projections[v] = initial_vertex_projections[v]
        continue
      nn_array = [(nn, 1-dist) for (nn, dist) in knn_graph[v]]
      
      nominator = MulScalarByVector(args.nu, uniform_pos)      
      denominator = args.nu
      for nn, weight in nn_array:
        nn_pos_vector = current_projections.get(nn, uniform_pos)
        nominator = AddVector(nominator, MulScalarByVector(weight, nn_pos_vector))
        denominator += weight
      new_projections[v] = MulScalarByVector(1/denominator, nominator)
    current_projections = new_projections

if __name__ == '__main__':
  main()
