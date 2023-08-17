#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn

from torch_geometric.nn import GINEConv
from models_lib.gnn_block import GraphNorm, MeanPool, GraphPool
from models_lib.compound_encoder import AtomEmbedding, BondEmbedding, BondFloatRBF, BondAngleFloatRBF


class GeoGNNBlock(nn.Module):
    """
    GeoGNN Block
    """

    def __init__(self, embed_dim, dropout_rate, last_act, device):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GINEConv(nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
                                          nn.Linear(embed_dim * 2, embed_dim))).to(device)
        self.norm = nn.LayerNorm(embed_dim).to(device)
        self.graph_norm = GraphNorm(device).to(device)
        if last_act:
            self.act = nn.ReLU().to(device)
        self.dropout = nn.Dropout(p=dropout_rate).to(device)

    def forward(self, graph, node_hidden, edge_hidden, node_id, edge_id):
        """tbd"""
        out = self.gnn(node_hidden, graph.edge_index, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(graph, out, node_id, edge_id)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out


class GeoGNNModel(nn.Module):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, args, model_config={}, device=None):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config['bond_angle_float_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim, device=device)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim, device=device)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim, device=device)

        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                BondEmbedding(self.bond_names, self.embed_dim, device=device))
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.bond_float_names, self.embed_dim, device=device))
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim, device=device))
            self.atom_bond_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), device=device))
            self.bond_angle_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), device=device))

        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = GraphPool(pool_type=self.readout)

        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)
        print('[GeoGNNModel] atom_names:%s' % str(self.atom_names))
        print('[GeoGNNModel] bond_names:%s' % str(self.bond_names))
        print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, node_id, edge_id):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.x.T)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_attr.T)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_attr.T[len(self.bond_names):])

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                atom_bond_graph,
                node_hidden_list[layer_id],
                edge_hidden_list[layer_id], node_id[0], edge_id[0])

            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_attr.T)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_attr.T[len(self.bond_names):])
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_attr.T)
            edge_hidden = self.bond_angle_block_list[layer_id](
                bond_angle_graph,
                cur_edge_hidden,
                cur_angle_hidden, node_id[1], edge_id[1])
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        # graph_repr = self.graph_pool(atom_bond_graph, node_repr, node_id[0], edge_id[0])
        return node_repr, edge_repr

