
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.features import BatchMolGraph
from chemprop.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, bias, depth, dropout, activation, device):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.device = device
        self.layers_per_message = 1

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout).to(device)

        # Activation
        self.act_func = get_activation_function(activation).to(device)

        # Input
        self.W_i_atom = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias).to(device)
        self.W_i_bond = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias).to(device)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias).to(device)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias).to(device)

        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size).to(device)

        self.gru = BatchGRU(self.hidden_size).to(device)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias).to(device)


    def forward(self, mol_graph: BatchMolGraph, features_batch=None, batch_mask=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = (f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device),
                                              b2a.to(self.device), b2revb.to(self.device))
        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens[1:]  # B x H


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message

