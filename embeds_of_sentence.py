# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
# from main_model import BERT_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model,load_label_list
import utils
from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
from dataloader.DataLoader import DataLoader_r as DataLoader
import fire
import numpy as np
# from tensorboardX import SummaryWriter
from calculate_F1 import *
from main_model import *
from dataloader.DataLoader import DataSetRewrite
import logging
import time
import matplotlib.pyplot as plt


config = Config()
print('当前设置为:\n', config)
# if config.use_cuda:
#     torch.cuda.set_device(config.gpu)
print('loading corpus')
# vocab = load_vocab(config.vocab)
label_dic = load_vocab(config.label_file)
tagset_size = len(label_dic)
# train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
# dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

train_dataset = DataSetRewrite(config.train_file, config.max_length, label_dic)
dev_dataset = DataSetRewrite(config.dev_file, config.max_length, label_dic)
test_dataset = DataSetRewrite(config.dev_file, config.max_length, label_dic)