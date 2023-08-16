#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pgl


import numpy as np
from copy import deepcopy
import hashlib
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from rdkit.Chem import AllChem
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from torch_geometric.data import Data, DataLoader, Batch, Dataset
from pgl import graph
def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(
        g,
        superedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0,
        subgraph_num=None,
        version='gem'):
    """tbd"""

    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)

        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value
    return [g, superedge_g, target_atom_indices, target_labels]


def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    def _add_item(
            node_i_indices, node_j_indices, node_k_indices, bond_angles,
            node_i_index, node_j_index, node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b1)
            if a0 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b0)
            if a1 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b1)
            if a1 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]

class GeoPredTransformFn(object):
    """Gen features for downstream model"""
    def __init__(self, pretrain_tasks, mask_ratio):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """
        node_i, node_j, node_k, bond_angles = \
                get_pretrain_bond_angle(data['edges'], data['atom_pos'])
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles

        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])

        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])
        return data

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data
        print('smiles', smiles)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        data['smiles'] = smiles
        # data = self.prepare_pretrain_task(data)
        return data


class GeoPredCollateFn(object):
    """tbd"""

    def __init__(self,
                 atom_names,
                 bond_names,
                 bond_float_names,
                 bond_angle_float_names,
                 pretrain_tasks,
                 mask_ratio,
                 Cm_vocab):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.Cm_vocab = Cm_vocab
        self.bond_angle_float_names = bond_angle_float_names

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):
        """tbd"""
        atom_bond_graph_list = []
        bond_angle_graph_list = []

        node_count = 0
        for data in batch_data_list:
            N = len(data[self.atom_names[0]])
            E = len(data['edges'])

            ab_g = Data(num_nodes=N, edge_index=data['edges'],
                        x=torch.FloatTensor(np.stack([data[name] for name in self.atom_names]).T),
                        edge_attr=torch.FloatTensor(np.stack([data[name] for name in self.bond_names + self.bond_float_names])).T)
            ba_g = Data(num_nodes=E, edges_index=data['BondAngleGraph_edges'],
                        edge_attr=torch.FloatTensor(np.stack([data[name] for name in self.bond_angle_float_names])).T)
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)

            node_count += N

        graph_dict = {}
        feed_dict = {}

        atom_bond_graph = Batch.from_data_list(atom_bond_graph_list)
        graph_dict['atom_bond_graph'] = atom_bond_graph

        bond_angle_graph = Batch.from_data_list(bond_angle_graph_list)
        graph_dict['bond_angle_graph'] = bond_angle_graph


        return graph_dict, feed_dict