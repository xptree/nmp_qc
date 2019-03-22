#!/usr/bin/env python
# encoding: utf-8
# File Name: test_aichemy.py
# Author: Jiezhong Qiu
# Create Time: 2019/03/22 12:57
# TODO:

import networkx as nx

import torch.utils.data as data
import numpy as np
import argparse

import datasets.utils as utils
from datasets.aichemy import AIChemy

import time
import os,sys

import torch

parser = argparse.ArgumentParser(description='QM9 Object.')
# Optional argument
parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['GraphReader/'])

args = parser.parse_args()
root = args.root[0]

files = [f for f in os.listdir(root) \
        if os.path.isfile(os.path.join(root, f)) \
        and os.path.splitext(f)[-1] == ".json"]

test = AIChemy(root, files, vertex_transform=utils.qm9_nodes,
        e_representation="raw_distance")

print(len(test))


start = time.time()
print(utils.get_graph_stats(test, ['target_mean', 'target_std']))
end = time.time()
print('Time Statistics Par')
print(end - start)
