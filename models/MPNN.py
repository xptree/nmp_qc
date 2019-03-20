#!/usr/bin/python
# -*- coding: utf-8 -*-

from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

import torch
import torch.nn as nn
from torch.autograd import Variable
import time

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class MPNN(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n,
            hidden_state_size,
            message_size,
            n_layers,
            l_target,
            type='regression',
            readout='s2s',
            edge_hidden_dim=50,
            set2set_comps=12,
            hidden_dim=200
            ):

        super(MPNN, self).__init__()

        # Define message
        self.m = nn.ModuleList([
            MessageFunction('mpnn',
                args={'edge_feat': in_n[1],
                    'in': hidden_state_size,
                    'out': message_size,
                    'edge_num_layers': n_layers,
                    'edge_hidden_dim': edge_hidden_dim,
                    }
                )
            ])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('mpnn',
                                               args={'in_m': message_size,
                                                     'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('mpnn_s2s' if readout == 's2s' else 'mpnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target,
                                       'set2set_comps': set2set_comps,
                                       'hidden_dim': hidden_dim
                                       })

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers

    def forward(self, g, h_in, e):

        # h_in is of size bsz x n x node_feat
        node_mask = torch.sum(h_in, dim=-1) == 0.0 # bsz x n

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        #  m_time, u_time, r_time = 0, 0, 0
        # Layer
        for t in range(0, self.n_layers):
            e_aux = e.view(-1, e.size(3))

            h_aux = h[t].view(-1, h[t].size(2))

            #  start = time.time()
            m = self.m[0].forward(h[t], h_aux, e_aux)
            #  m_time += time.time() - start

            m = m.view(h[0].size(0), h[0].size(1), -1, m.size(1))

            # Nodes without edge set message to 0
            m = torch.unsqueeze(g, 3).expand_as(m) * m

            m = torch.squeeze(torch.sum(m, 1))

            #  start = time.time()
            h_t = self.u[0].forward(h[t], m)
            #  u_time += time.time() - start

            # Delete virtual nodes
            #h_t = (torch.sum(h_in, 2).expand_as(h_t) > 0).type_as(h_t) * h_t
            h_t = (torch.sum(h_in, 2).unsqueeze(dim=-1).expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        #  start = time.time()
        res = self.r.forward(h, node_mask)
        #  r_time += time.time() - start

        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        #  print("m_time=%f u_time=%f r_time=%f" % (m_time, u_time, r_time))
        return res
