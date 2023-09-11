#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: main.py
@time: 2023/8/13 20:05
@desc:
'''
import os
import copy
import torch
import json
import logging
import warnings

import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from parser_args import get_args
from tensorboardX import SummaryWriter
from chemprop.data import StandardScaler
from utils.dataset import Seq2seqDataset, get_data, split_data, MoleculeDataset, InMemoryDataset, load_npz_to_data_list
from utils.evaluate import eval_rocauc, eval_rmse
from torch.utils.data import BatchSampler, RandomSampler, DataLoader
from build_vocab import WordVocab
from chemprop.nn_utils import NoamLR
from chemprop.features import mol2graph, get_atom_fdim, get_bond_fdim
from chemprop.data.utils import get_class_sizes
from models_lib.multi_modal import Multi_modal
from featurizers.gem_featurizer import GeoPredTransformFn

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
warnings.filterwarnings('ignore')


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


def prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, device):
    edge_batch1, edge_batch2 = [], []
    geo_gen = geo_data.get_batch(idx)
    node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
    for i in range(geo_gen[0].num_graphs):
        edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
    edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]
    # 2D data
    mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
    smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
    gnn_batch = mol2graph(smiles_batch, args)
    batch_mask_seq, batch_mask_gnn = list(), list()
    for i, (smile, mol) in enumerate(zip(smiles_batch, mol_batch.mols())):
        batch_mask_seq.append(torch.ones(len(smile), dtype=torch.long).to(device) * i)
        batch_mask_gnn.append(torch.ones(mol.GetNumAtoms(), dtype=torch.long).to(device) * i)
    batch_mask_seq = torch.cat(batch_mask_seq)
    batch_mask_gnn = torch.cat(batch_mask_gnn)
    mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
    targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
    return seq_data[idx], seq_mask[idx], batch_mask_seq, gnn_batch, features_batch, batch_mask_gnn, geo_gen, \
           node_id_all, edge_id_all, mask, targets


def train(args, model, optimizer, scheduler, train_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device):
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    for idx in tqdm(train_idx_loader):
        model.zero_grad()
        # 3D data
        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, device)
        x_list, preds = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all)
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, targets, mask)
        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        all_loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    return all_loss.item(), lab_loss.item(), cl_loss.item(), model


@torch.no_grad()
def val(args, model, scaler, val_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device):
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    y_true = []
    y_pred = []
    for idx in val_idx_loader:
        # 3D data

        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, device)
        x_list, preds = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all)
        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu()).astype(np.float64)).to(device)
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, targets, mask, args.cl_loss)
        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        y_true.append(targets)
        y_pred.append(preds)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    # print('result:', result)
    return result, all_loss.item(), lab_loss.item(), cl_loss.item(), model


@torch.no_grad()
def test(args, model, scaler, test_idx_loader, seq_data, seq_mask, gnn_data, geo_data, device):
    y_true = []
    y_pred = []
    for idx in test_idx_loader:
        # 3D data

        seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch, geo_gen, node_id_all, \
        edge_id_all, mask, targets = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, device)
        x_list, preds = model(seq_batch, seq_batch_mask, seq_batch_batch, gnn_batch, features_batch, gnn_batch_batch,
                              geo_gen, node_id_all, edge_id_all)
        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu()).astype(np.float64))
        y_true.append(targets)
        y_pred.append(preds)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    # print('result:', result)
    return result


def main(args):
    logs_dir = './LOG/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/'.format(args.dataset, args.lr, args.cl_loss, args.cl_loss_num,
                                                                 args.pro_num, args.pool_type, args.fusion, args.epochs,
                                                                 args.norm, args.gnn_hidden_dim, args.batch_size)
    logs_file1 = logs_dir + "Train_{}".format(args.seed)
    writer = SummaryWriter(log_dir=logs_file1)
    logs_file = logs_dir + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_Train_{}.log".format(args.dataset, args.lr, args.cl_loss,
                                                                                  args.cl_loss_num, args.pro_num,
                                                                                  args.pool_type, args.fusion,
                                                                                  args.epochs, args.norm,
                                                                                  args.gnn_hidden_dim, args.batch_size,
                                                                                  args.seed)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger = logging.getLogger(logs_file)
    fh = logging.FileHandler(logs_file)
    logger.addHandler(fh)
    logging.basicConfig(level=logging.DEBUG,
                        filename=logs_file,
                        filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # device init
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
        logger.info("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        logger.info("Device set to : cpu")

    logger.info("lr:" + str(args.lr) + ", cl_loss:" + str(args.cl_loss) + ", cl_loss_num:" + str(
        args.cl_loss_num) + ", pro_num:" + str(args.pro_num) + ", pool_type:" + str(
        args.pool_type) + ", gnn_hidden_dim:" + str(args.gnn_hidden_dim) + ", batch_size:" + str(
        args.batch_size) + ", norm:" + str(args.norm) + ", fusion:" + str(args.fusion))

    # gnn data
    data_path = 'data/{}.csv'.format(args.dataset)
    # data_3d = load_smiles_to_dataset(args.data_path_3d)
    datas, args.seq_len = get_data(path=data_path, args=args)
    # datas = MoleculeDataset(datas[0:8])
    args.output_dim = args.num_tasks = datas.num_tasks()
    args.gnn_atom_dim = get_atom_fdim(args)
    args.gnn_bond_dim = get_bond_fdim(args) + (not args.atom_messages) * args.gnn_atom_dim
    args.features_size = datas.features_size()
    # data split
    train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes,
                                                 seed=args.seed, args=args)
    train_idx = [data.idx for data in train_data]
    val_idx = [data.idx for data in val_data]
    test_idx = [data.idx for data in test_data]
    # seq data process
    smiles = datas.smiles()
    vocab = WordVocab.load_vocab('./data/{}_vocab.pkl'.format(args.dataset))
    args.seq_input_dim = args.vocab_num = len(vocab)
    seq = Seq2seqDataset(list(np.array(smiles)), vocab=vocab, seq_len=args.seq_len, device=device)
    seq_data = torch.stack([temp[1] for temp in seq])

    # 3d data process
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    data_3d = InMemoryDataset(datas.smiles())
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    if not os.path.exists('./data/{}/'.format(args.dataset)):
        data_3d.transform(transform_fn, num_workers=1)
        data_3d.save_data('./data/{}/'.format(args.dataset))
    else:
        data_3d = data_3d._load_npz_data_path('./data/{}/'.format(args.dataset))
        data_3d = InMemoryDataset(data_3d)

    train_sampler = RandomSampler(train_idx)
    val_sampler = BatchSampler(val_idx, batch_size=args.batch_size, drop_last=False)
    test_sampler = BatchSampler(test_idx, batch_size=args.batch_size, drop_last=False)
    train_idx_loader = DataLoader(train_idx, batch_size=args.batch_size, sampler=train_sampler)
    data_3d.get_data(device)

    #
    seq_mask = torch.zeros(len(datas), args.seq_len).bool().to(device)
    for i, smile in enumerate(smiles):
        seq_mask[i, 1:1 + len(smile)] = True
    # task information
    if args.task_type == 'class':
        class_sizes = get_class_sizes(datas)
        for i, task_class_sizes in enumerate(class_sizes):
            print(f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
    if args.task_type == 'reg':
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
        for (id, value) in zip(train_idx, scaled_targets):
            datas[id].set_targets(value)
    else:
        scaler = None

    # Multi Modal Init
    args.seq_hidden_dim = args.gnn_hidden_dim
    args.geo_hidden_dim = args.gnn_hidden_dim
    model = Multi_modal(args, compound_encoder_config, device)
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    schedule = NoamLR(optimizer=optimizer, warmup_epochs=[args.warmup_epochs], total_epochs=[args.epochs],
                      steps_per_epoch=len(train_idx) // args.batch_size, init_lr=[args.init_lr],
                      max_lr=[args.max_lr], final_lr=[args.final_lr], )
    ids = list(range(len(train_data)))
    best_result = None
    best_test = None
    best_epoch = 0
    torch.backends.cudnn.enabled = False
    logger.info('train model ...')
    for epoch in range(args.epochs):
        np.random.shuffle(ids)
        # train
        train_all_loss, train_lab_loss, train_cl_loss, model = train(args, model, optimizer, schedule, train_idx_loader,
                                                                     seq_data, seq_mask, datas, data_3d, device)
        logger.info("epoch:" + str(epoch) + ", all_loss:" + str(train_all_loss) + ", lab_loss:" + str(
            train_lab_loss) + ", cl_loss:" + str(train_cl_loss))
        # val
        model.eval()
        val_result, val_all_loss, val_lab_loss, val_cl_loss, model = val(args, model, scaler, val_sampler, seq_data,
                                                                         seq_mask, datas, data_3d, device)
        writer.add_scalars(main_tag='loss',
                           tag_scalar_dict={'train_all_loss': train_all_loss, 'train_lab_loss': train_lab_loss,
                                            'train_cl_loss': train_cl_loss, 'val_all_loss': val_all_loss,
                                            'val_lab_loss': val_lab_loss, 'val_cl_loss': val_cl_loss},
                           global_step=int(epoch + 1))

        writer.add_scalars(main_tag='result', tag_scalar_dict={'val_result': val_result}, global_step=int(epoch + 1))
        if best_result is None or (best_result < val_result and args.task_type == 'class') or \
                (best_result > val_result and args.task_type == 'reg'):
            save_model = copy.deepcopy(model)
            best_result = val_result
            logger.info("--min_val_loss:" + str(val_all_loss) + ", val_result:" + str(val_result))
        # if not (epoch + 1) % 5:
        #     save_model.eval()
            result = test(args, save_model, scaler, test_sampler, seq_data, seq_mask, datas, data_3d, device)
            logger.info("*************result:" + str(result) + "**************\n")
            if best_test is None or best_test < result:
                best_test = result
                best_epoch = epoch
        torch.cuda.empty_cache()

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)



if __name__ == "__main__":
    arg = get_args()
    main(arg)
