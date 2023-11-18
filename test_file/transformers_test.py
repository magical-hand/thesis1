import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 在此我指定使用2号GPU，可根据需要调整
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset
import os

def ncbi_disease():
    train_dataset = load_dataset("ncbi_disease",split="train")
    dev_dataset = load_dataset("ncbi_disease", split="train[400:500]")
    test_dataset = load_dataset("ncbi_disease", split="test")
    # print(train_dataset.features)
    print(dev_dataset)
    print(test_dataset)

    print(train_dataset['ner_tags'])
    A = train_dataset.info.features['ner_tags'].feature.names
    token_num=0
    for sentence in train_dataset['ner_tags']:
        token_num+=len(sentence)

    print(token_num,'////')

def conll_2003():
    train_dataset = load_dataset("conll2003", split="train")
    # dev_dataset = load_dataset("ncbi_disease", split="train[400:500]")
    # test_dataset = load_dataset("ncbi_disease", split="test")
    # print(train_dataset.features)
    # print(dev_dataset)
    # print(test_dataset)
    #
    # print(train_dataset['ner_tags'])
    #
    # print(test_dataset.info.features['ner_tags'],'???????')

    A=train_dataset.info.features['ner_tags'].feature.names

    # print(train_dataset.features)

    token_num = 0
    for sentence in train_dataset['ner_tags']:
        token_num += len(sentence)

    print(token_num, '////')


def conllpp():
    train_dataset = load_dataset("conllpp",split="train")
    dev_dataset = load_dataset("ncbi_disease", split="train[400:500]")
    test_dataset = load_dataset("ncbi_disease", split="test")
    # print(train_dataset.features)
    # print(dev_dataset)
    # print(test_dataset)
    #
    # print(train_dataset['ner_tags'])
    A = train_dataset.info.features['ner_tags'].feature.names
    token_num=0
    for sentence in train_dataset['ner_tags']:
        token_num+=len(sentence)

    print(token_num,'////')


def ontonotes(train_dataset,set_name,sample_tuple=None,file_name_prefix=None):

    entity_list=train_dataset[set_name].info.features['sentences'][0]['named_entities'].feature.names
    trainset_length=train_dataset.num_rows['train']
    if set_name=='validation':
        set_name1='testa.txt'
    if set_name=='train':
        set_name1='train.txt'
    if set_name=='test':
        set_name1='testb.txt'
    if sample_tuple:
        set_name1='set'+str(file_name_prefix)+'/train.txt'
        step=0
        with open(r'ACE_DATASET/'+set_name1,'w',encoding='utf-8') as w:
            for data_part in train_dataset[set_name]['sentences']:
                for line in data_part:
                    if step in sample_tuple:

                        for j in range(len(line['words'])):
                            word=line['words'][j]
                            pos=line['pos_tags'][j]
                            chunk=line['word_senses'][j]
                            tag=entity_list[line['named_entities'][j]]
                            w.write(str(word)+' '+str(pos)+' '+str(chunk)+' '+str(tag)+'\n')
                        w.write('\n')
                    step+=1
    else:
        with open(r'ACE_DATASET/'+set_name1,'w',encoding='utf-8') as w:
            for data_part in train_dataset[set_name]['sentences']:
                for line in data_part:
                    for j in range(len(line['words'])):
                        word=line['words'][j]
                        pos=line['pos_tags'][j]
                        chunk=line['word_senses'][j]
                        tag=entity_list[line['named_entities'][j]]
                        w.write(str(word)+' '+str(pos)+' '+str(chunk)+' '+str(tag)+'\n')
                    w.write('\n')


def ontonotes_2(train_dataset,set_name,sample_tuple=None,file_name_prefix=None):

    entity_list=train_dataset[set_name].info.features['sentences'][0]['named_entities'].feature.names
    # trainset_length=train_dataset.num_rows['train']
    if set_name=='validation':
        set_name1='testa.txt'
    if set_name=='train':
        set_name1='train.txt'
    if set_name=='test':
        set_name1='testb.txt'
    if sample_tuple:
        set_name1='set'+str(file_name_prefix)+'/train.txt'
        step=0

        path=r'D_ACE_DATASET/set'+str(file_name_prefix)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(r'D_ACE_DATASET/'+set_name1,'w',encoding='utf-8') as w:
            for data_part in train_dataset[set_name]['sentences']:

                for line in data_part:
                    if step in sample_tuple:
                        tag_text=[entity_list[i] for i in line['named_entities']]

                        tag_text=' '.join(tag_text)
                        word=' '.join(line['words'])
                        w.write(word+'|||'+tag_text+'\n')
                        # w.write('\n')
                    step+=1
                # part_num+=1
    else:
        part_num = 0
        entity_num_list = {}
        if not os.path.exists(r'D_ACE_DATASET'):
            os.makedirs(r'D_ACE_DATASET')
        with open(r'D_ACE_DATASET/'+set_name1,'w',encoding='utf-8') as w:
            for data_part in train_dataset[set_name]['sentences']:
                entity_num_list[part_num] = []
                for line in data_part:
                    tag_text = [entity_list[i] for i in line['named_entities']]
                    for tag in tag_text:
                        if tag not in entity_num_list[part_num]:
                            entity_num_list[part_num].append(tag)
                    tag_text = ' '.join(tag_text)
                    word = ' '.join(line['words'])
                    w.write(word +'|||'+ tag_text + '\n')
                part_num+=1
    return 0

# conll_2003()
# # ncbi_disease()
# conllpp()
def transfor_data(ontonotes):
    train_dataset = load_dataset("conll2012_ontonotesv5","english_v4")
    for set_name in ['train','test','validation']:
        if set_name=='train':
            set_name1='train.txt'
            trainset_length = train_dataset.num_rows['train']
            rand_tuple5 = random.sample(range(0, trainset_length), trainset_length // 3)
            rand_tuple1 = random.sample(range(0, trainset_length), trainset_length // 9)
            ontonotes(train_dataset, set_name)
            ontonotes(train_dataset,set_name,rand_tuple5,3)
            ontonotes(train_dataset,set_name,rand_tuple1,1)
        else:
            ontonotes(train_dataset,set_name)

transfor_data(ontonotes_2)