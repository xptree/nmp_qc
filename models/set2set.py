#!/usr/bin/env python
# encoding: utf-8
# File Name: set2set.py
# Author: Jiezhong Qiu
# Create Time: 2019/03/19 14:55
# TODO:

import torch
import torch.nn.functional as F


class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper
    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})
        \alpha_{i} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)
                \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i} \mathbf{x}_i
        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,
    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.
    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, out_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = out_channels * 2
        self.out_channels = out_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=self.in_channels,
									hidden_size=self.out_channels,
                                  	num_layers=num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, mask):
        batch_size = x.size(0)
		# x is of shape (bsz x n x out_channels)
        # mask is of shape (bsz x n)

        h = (x.new_zeros((self.num_layers, batch_size, self.out_channels)),
             x.new_zeros((self.num_layers, batch_size, self.out_channels)))
        q_star = x.new_zeros(batch_size, self.in_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h) # 1 x bsz x out_channel
            q = q.squeeze(0) # bsz x out_channel
            e = torch.bmm(x, q.unsqueeze(-1)).squeeze(-1) # bsz x n
            e.masked_fill_(mask, float('-inf')) # bsz x n
            a = F.softmax(e, dim=1).unsqueeze(1) # bsz x 1 x n
            r = torch.bmm(a, x).squeeze(1) # bsz x out_channel
            q_star = torch.cat([q, r], dim=-1) # bsz x in_channel

        return q_star # bs x (2*out_channel)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
