# from typing import List
# #from flair.datasets import conll_03_new, CONLL_03_DUTCH, CONLL_03_SPANISH, CONLL_03_GERMAN
import time

from flair import embeddings as Embeddings
# print(time.time())
# import torch
# from torch.utils.data.dataset import ConcatDataset
# from flair.training_utils import store_embeddings
# # initialize sequence tagger
# from pathlib import Path
# import argparse
# import yaml
# # from . import logging
# import pdb
# import copy
from opration import embbedings_config as embeddings
# print(time.time(),'?')
# from flair import datasets as datasets
# from list_data import ListCorpus

class embeddings_generater():
    def __init__(self,datasets_config):
        self.full_corpus = {'ner': 'CONLL_03_GERMAN:conll_03_new:CONLL_03_DUTCH:CONLL_03_SPANISH',
                        'upos': 'UD_GERMAN:UD_ENGLISH:UD_FRENCH:UD_ITALIAN:UD_DUTCH:UD_SPANISH:UD_PORTUGUESE:UD_CHINESE'}
        # datasets_config= {
        #         'data_folder': r'C:\Users\Administrator\Desktop\ACE-main\datasets\conll_03_english',
        #         'column_format':{
        #           0: 'text',
        #           1: 'pos',
        #           2: 'chunk',
        #           3: 'ner'},
        #         'tag_to_bioes': 'ner'}
        # current_dataset = datasets.ColumnCorpus(**datasets_config)
        embedding_list=[]
        # corpus_list={'train':[],'dev':[],'test':[]}
        # corpus_list['train'].append(current_dataset.train)
        # corpus_list['dev'].append(current_dataset.dev)
        # corpus_list['test'].append(current_dataset.test)
        # corpus_list['targets'] = ['ColumnCorpus-1']
        # self.corpus: ListCorpus = ListCorpus(**corpus_list)
        # self.dateset=datasets.ColumnCorpus(datasets_config)
        # self.tokens = self.corpus.get_train_full_tokenset(-1, min_freq=-1)
        # self.lemmas = self.corpus.get_train_full_tokenset(-1, min_freq=-1, attr='lemma')[0]
        # self.postags = self.corpus.get_train_full_tokenset(-1, min_freq=-1, attr='pos')[0]

        for embedding in embeddings:
            if 'FastWordEmbeddings' in embedding:
                embedding_list.append(
                    getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))
                word_map = embedding_list[-1].vocab
            # elif 'LemmaEmbeddings' in embedding:
            #     embedding_list.append(
            #         getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding], vocab=self.lemmas))
            #     lemma_map = embedding_list[-1].lemma_dictionary
            # elif 'POSEmbeddings' in embedding:
            #     embedding_list.append(
            #         getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding], vocab=self.postags))
            #     postag_map = embedding_list[-1].pos_dictionary
            elif 'FastCharacterEmbeddings' in embedding:
                embedding_list.append(
                    getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))
                char_map = embedding_list[-1].char_dictionary
            else:
                embedding_list.append(getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))

            self.embeddings_list=embedding_list
