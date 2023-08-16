#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: multi_modal.py
@time: 2022/3/28 20:05
@desc:
'''
import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, GlobalAttention
from models_lib.gnn_model import MPNEncoder
from models_lib.gem_model import GeoGNNModel
from models_lib.seq_model import TrfmSeq2seq

loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"), 'reg': nn.MSELoss(reduction="none")}


class Global_Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):

        return self.at(x, batch)

class WeightFusion(nn.Module):

    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return sum([input[i]*weight for i, weight in enumerate(self.weight[0][0])]) + self.bias


class Multi_modal(nn.Module):
    def __init__(self, args, compound_encoder_config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        # CMPNN
        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=args.gnn_hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)
        # Transformer
        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=args.seq_hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers, dropout=args.dropout,
                                       vocab_num=args.vocab_num, device=device, recons=args.recons).to(self.device)
        # Geometric GNN
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)

        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = nn.Sequential(nn.Linear(args.gnn_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_geo = nn.Sequential(nn.Linear(args.geo_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = self.pro_seq
            self.pro_geo = self.pro_seq

        self.entropy = loss_type[args.task_type]

        if args.pool_type == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = Global_Attention(args.seq_hidden_dim).to(self.device)

        # Fusion
        fusion_dim = args.gnn_hidden_dim * self.graph + args.seq_hidden_dim * self.sequence + \
                     args.geo_hidden_dim * self.geometry
        if self.args.fusion == 3:
            fusion_dim /= (self.graph + self.sequence + self.geometry)
            self.fusion = WeightFusion(self.graph + self.sequence + self.geometry, fusion_dim, device=self.device)
        elif self.args.fusion == 2 or self.args.fusion == 0:
            fusion_dim = args.seq_hidden_dim

        self.dropout = nn.Dropout(args.dropout)

        # MLP Classifier
        self.output_layer = nn.Sequential(nn.Linear(int(fusion_dim), int(fusion_dim)), nn.ReLU(), nn.Dropout(args.dropout),
                                          nn.Linear(int(fusion_dim), args.output_dim)).to(self.device)


    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all):

        x_list = list()
        cl_list = list()
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            if self.args.norm:
                x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))
            else:
                x_list.append(graph_gnn_x)
            cl_list.append(self.pro_gnn(graph_gnn_x))

        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            if self.args.norm:
                x_list.append(F.normalize(graph_seq_x, p=2, dim=1))
            else:
                x_list.append(graph_seq_x)
            cl_list.append(self.pro_seq(graph_seq_x))

        if self.geometry:
            node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            if self.args.norm:
                x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
            else:
                x_list.append(graph_geo_x)
            cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))


        if self.args.fusion == 1:
            molecule_emb = torch.cat([temp for temp in x_list], dim=1)
        elif self.args.fusion == 2:
            molecule_emb = x_list[0].mul(x_list[1]).mul(x_list[2])
        elif self.args.fusion == 3:
            molecule_emb = self.fusion(torch.stack(x_list, dim=0))
        else:
            molecule_emb = torch.mean(torch.cat(x_list), dim=0, keepdim=True)

        if not self.args.norm:
            molecule_emb = self.dropout(molecule_emb)

        pred = self.output_layer(molecule_emb)
        return cl_list, pred

    def label_loss(self, pred, label, mask):
        loss_mat = self.entropy(pred, label)
        return loss_mat.sum() / mask.sum()

    def cl_loss(self, x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss1).mean()
        return loss


    def loss_cal(self, x_list, pred, label, mask, alpha=0.08):
        loss1 = self.label_loss(pred, label, mask)
        loss2 = torch.tensor(0, dtype=torch.float).to(self.device)
        modal_num = len(x_list)
        for i in range(modal_num):
            loss2 += self.cl_loss(x_list[i], x_list[i-1])

        return loss1 + alpha * loss2, loss1, loss2


