#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
qm9.py:

Usage:

"""

# Networkx should be imported before torch
import networkx as nx

import torch.utils.data as data
import numpy as np
import argparse

import datasets.utils as utils

import time
import os,sys

import torch

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)


from GraphReader.graph_reader import aichemy_graph_reader

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

_label_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                'zpve', 'U0', 'U', 'H', 'G', 'Cv']

class AIChemy(data.Dataset):

    # Constructor
    def __init__(self, root_path, ids, vertex_transform=utils.qm9_nodes, edge_transform=utils.qm9_edges,
                 target_transform=None, e_representation='raw_distance', labels=None):
        self.root = root_path
        self.ids = ids
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation
        labels = labels or _label_names

        if isinstance(labels, str):
            labels = [labels, ]

        self.labels_id = np.array([_label_names.index(x) for x in labels])

    def __getitem__(self, index):
        g, target = aichemy_graph_reader(os.path.join(self.root, self.ids[index]),
                labels_id=self.labels_id)
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (g, h, e), target

    def __len__(self):
        return len(self.ids)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform

if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='QM9 Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['../data/GraphReader/'])

    args = parser.parse_args()
    root = args.root[0]

    files = [f for f in os.listdir(root) \
            if os.path.isfile(os.path.join(root, f)) \
            and os.path.splitext(f)[-1] == ".json"]

    data = AIChemy(root, files, vertex_transform=utils.qm9_nodes, edge_transform=lambda g: utils.qm9_edges(g, e_representation='raw_distance'))

    print(len(data))


    start = time.time()
    print(utils.get_graph_stats(data, 'degrees'))
    end = time.time()
    print('Time Statistics Par')
    print(end - start)
