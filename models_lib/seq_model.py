#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: seq_model.py
@time: 2023/8/13 20:05
@desc:
'''

import math
import torch

from torch import nn
from torch.nn import functional as F

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class TrfmSeq2seq(nn.Module):
    def __init__(self, input_dim, hidden_size, num_head, n_layers, dropout, vocab_num, device, recons=False):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = input_dim
        self.hidden_size = hidden_size
        self.embed = nn.LSTM(input_size=input_dim, hidden_size=int(hidden_size/2), bidirectional=True, batch_first=True,
                             num_layers=3, dropout=dropout)
        self.recons = recons
        self.device = device

        self.vocab_num = vocab_num
        transformer = nn.Transformer(d_model=hidden_size, nhead=num_head, num_encoder_layers=n_layers,
                                     num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.out = nn.Linear(hidden_size, input_dim)

    def forward(self, src):
        # src: (T,B)\sout{}
        loss = 0
        embedded, _ = self.embed(src.to(self.device))  # (T,B,H)
        hidden = self.encoder(embedded)  # (T,B,H)
        if self.recons:
            out = self.decoder(embedded, hidden)
            out = self.out(out)  # (T,B,V)
            out = F.log_softmax(out, dim=-1)  # (T,B,V)
            loss = self.recon_loss(out, src.to(self.device), self.vocab_num)
        return loss, hidden  # (T,B,V)

    def recon_loss(self, output, target, vocab_max_num):
        loss = F.nll_loss(output.view(-1, vocab_max_num), target.contiguous().view(-1), ignore_index=PAD)
        return loss

