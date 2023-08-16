#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: data3d.py
@time: 2023/8/13 20:05
@desc:
'''
import copy

import torch
from parser_args import get_args
import numpy as np
import json

from utils.dataset import Seq2seqDataset, get_data, split_data, MoleculeDataset

from chemprop.features import mol2graph, get_atom_fdim, get_bond_fdim
from utils.build_vocab import WordVocab
from chemprop.data.utils import split_data, get_class_sizes

from utils.dataset import InMemoryDataset, load_npz_to_data_list

from featurizers.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn


PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))


def load_smiles_to_dataset(data_path):
    """tbd"""
    data_list = []
    with open(data_path, 'r') as f:
        tmp_data_list = [line.strip() for line in f.readlines()]
        tmp_data_list = tmp_data_list[1:]
    data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list)
    return dataset

def main(args):

    # device init
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # gnn data
    data_path = 'data/{}.csv'.format(args.dataset)
    # data_3d = load_smiles_to_dataset(args.data_path_3d)
    datas, args.seq_len = get_data(path=data_path, args=args)
    # datas = MoleculeDataset(datas[0:128])
    args.output_dim = args.num_tasks = datas.num_tasks()
    args.features_size = datas.features_size()
    args.gnn_atom_dim = get_atom_fdim(args)
    args.gnn_bond_dim = get_bond_fdim(args) + (not args.atom_messages) * args.gnn_atom_dim
    # seq data process
    vocab = WordVocab.load_vocab('./data/{}_vocab.pkl'.format(args.dataset))
    args.seq_input_dim = args.vocab_num = len(vocab)
    smiles = datas.smiles()
    train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes,
                                                 seed=args.seed, args=args)
    train_idx = [data.idx for data in train_data]
    val_idx = [data.idx for data in val_data]
    test_idx = [data.idx for data in test_data]

    seq_train = Seq2seqDataset(list(np.array(smiles)[train_idx]), vocab=vocab, seq_len=args.seq_len)
    seq_val = Seq2seqDataset(list(np.array(smiles)[val_idx]), vocab=vocab, seq_len=args.seq_len)
    seq_test = Seq2seqDataset(list(np.array(smiles)[test_idx]), vocab=vocab, seq_len=args.seq_len)

    #3d data process

    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate

    data_3d = InMemoryDataset(datas.smiles())
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    data_3d.transform(transform_fn, num_workers=1)
    data_3d.save_data('./data/{}/'.format(args.dataset))
    # data_3d = load_npz_to_data_list('./data/{}/part-000000.npz'.format(args.dataset))
    # data_3d = InMemoryDataset(data_3d)
    
if __name__=="__main__":
    arg = get_args()
    main(arg)