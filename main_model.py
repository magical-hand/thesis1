import configparser
import copy
from config import Config
from config import config_class

import torch
from flair.data import Sentence
# from flair.data_fetcher import NLPTaskDataFetcher
# from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
from flair import device
from flair.embeddings import StackedEmbeddings
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel
from crf import CRF
from torch.autograd import Variable
import torch
import torch.nn.functional as F
# import ipdb
from flair.embeddings import FlairEmbeddings,ELMoEmbeddings,WordEmbeddings,BytePairEmbeddings,TransformerWordEmbeddings,PooledFlairEmbeddings,CharacterEmbeddings
from opration import OPS
import torch.nn as nn
import numpy as np
import logging
from time import time
from allennlp.modules.elmo import Elmo, batch_to_ids
from embedding_generater import embeddings_generater


class BERT_LSTM_CRF(nn.Module):
    """
    bert_lstm_crf model
    """
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
        super(BERT_LSTM_CRF, self).__init__()
        # logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(message)s',
        #                     filename='time_recoder.log',
        #                     filemode='a',
        #                     level=logging.INFO)
        self.use_cuda=use_cuda
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.word_embeds = BertModel.from_pretrained(bert_config)

        self.word_embeds = MixedOp(use_cuda)
        self._initialize_alphas()
        word_embeds_len =self.word_embeds.stack_embedding.embedding_length
        # if "ELMoEmbeddings('small')" in OPS:
        #     word_embeds_len+=1024
        self.liner_1=nn.Linear(word_embeds_len,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim*2, tagset_size+2)
        self.tagset_size = tagset_size

    # def new(self):
    #     return self.copy_()

    def count_embeds_len(self):
        test_txt=['test']
        sentence=Sentence(test_txt)
        weight=[torch.randn(len(OPS))]
        embedding_cat=self.word_embeds(sentence,weight)
        return embedding_cat.shape[1]

    def _initialize_alphas(self):
        num_ops = len(self.word_embeds.embeddings_list)
        self._arch_parameters = Variable(torch.randn(num_ops))
        self._arch_parameters=self._arch_parameters.to(device)
        self._arch_parameters.requires_grad=True


    def arch_parameters(self):
        return self._arch_parameters

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        # time_1=time()



        # time_2=time()
        # logging.info('sentence1 time {}'.format(time_2-time_1))

        embeds = self.word_embeds(sentence,self.arch_parameters())

        # time_3 = time()
        # logging.info('sentence2 time {}'.format(time_3-time_2))

        batch_size = embeds.size(0)
        self.batch_sequence_length=embeds.shape[1]
        # hidden = self.rand_init_hidden(batch_size)
        # if embeds.is_cuda:
        #     hidden = (i.cuda() for i in hidden)
        #     hidden=list(hidden)
        # print(embeds.shape)

        embeds=self.liner_1(embeds)
        # embeds=torch.cat([self.liner_1(embeds[:,i,:]) for i in range(embeds.shape[1])])
        # print(embeds.shape)

        # time_4 = time()
        # logging.info('sentence3 time {}'.format(time_4 - time_3))

        lstm_out, hidden = self.lstm(embeds)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, self.batch_sequence_length, -1)

        # time_5 = time()
        # logging.info('sentence4 time {}'.format(time_5 - time_4))

        return lstm_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value

    # def _make_padded_tensor_for_batch(self, sentences) :
    #     names = self.word_embeds.stack_embedding.embeddings.get_names()
    #     lengths= [len(sentence.tokens) for sentence in sentences]
    #     longest_token_sequence_in_batch: int = max(lengths)
    #     embedding_length=self.word_embeds.stack_embedding.embeddings.embedding_length
    #     pre_allocated_zero_tensor = torch.zeros(
    #         embedding_length* longest_token_sequence_in_batch,
    #         dtype=torch.float,
    #         device=device,
    #     )
    #     all_embs = list()
    #     for sentence in sentences:
    #         all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
    #         nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
    #         cls_embedding=torch.ones(embedding_length)
    #         sep_embedding=cls_embedding*2
    #         all_embs=cls_embedding+all_embs+sep_embedding
    #         if nb_padding_tokens > 0:
    #             t = pre_allocated_zero_tensor[: embedding_length * nb_padding_tokens]
    #             all_embs.append(t)
    #
    #     sentence_tensor = torch.cat(all_embs).view(
    #         [
    #             len(sentences),
    #             longest_token_sequence_in_batch,
    #             self.word_embeds.stack_embedding.embeddings.embedding_length,
    #         ]
    #     )
    #     return torch.tensor(lengths, dtype=torch.long), sentence_tensor

class MixedOp(nn.Module):

    def __init__(self,use_cuda):
        super(MixedOp, self).__init__()
        config=Config()
        self.use_cuda=use_cuda
        # self._ops = []
        self.embeddings_list = embeddings_generater(config.dataset_config).embeddings_list

        # for primitive in OPS:  #PRIMITIVES中就是8个操作
        #     if primitive=="ELMoEmbeddings('small')":
        #         options_file = "model_ELMO/options.json"  # 配置文件地址
        #         weight_file = "model_ELMO/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"  # 权重文件地址
        #         # 这里的1表示产生一组线性加权的词向量。
        #         # 如果改成2 即产生两组不同的线性加权的词向量。
        #         self.elmo_model = Elmo(options_file, weight_file, 1, dropout=0)
        #         continue
        #     op = eval(primitive)    #OPS中存储了各种操作的函数
        #     self._ops.append(op)#把这些op都放在预先定义好的modulelist里
        self.stack_embedding=StackedEmbeddings(self.embeddings_list)

    def forward(self, x, weights):
        # x=Sentence(x)
        sentences = [Sentence(i) for i in x]
        self.stack_embedding.embed(sentences)
        # a=time()
        if config_class.all_embedding_used==False:
            weights = F.softmax(weights, dim=-1)
        else:
            weights=torch.ones_like(weights)
        weight_embedding_sentence=[]
        # print(len(self._ops),'qwerqwerq')
        for sentence in sentences:
            weight_embedding_token=[]
            for token in sentence.tokens:
                # print(token.embedding.shape,'??????')
                # print(token._embeddings[self._ops[1].name].device,weights.device)
                #print([token._embeddings[self._ops[i].name].shape for i in range(len(self._ops))])
                # print([(token._embeddings[self._ops[i].name].to(device)).shape for i in range(len(self._ops))])
                # print([(token._embeddings[self._ops[i].name].to(device)*weights[i]).shape for i in range(len(self._ops))])
                weight_embedding_token.append(torch.cat([token._embeddings[self.embeddings_list[i].name].to(device)*weights[i] for i in range(len(self.embeddings_list))]))
            weight_embedding_sentence.append(torch.stack(weight_embedding_token))
        # character_ids = batch_to_ids(x).to(device)
        # elmo_embeddings=self.elmo_model(character_ids)['elmo_representations'][0]*weights[-1]
            # print(weight_embedding_token[0].shape,'jkluoio')
        # print(time()-a,'>>>>>',time())
        result=torch.stack(weight_embedding_sentence).to(device)
        # result=torch.cat([result,elmo_embeddings],2)
        return result

    def _make_padded_tensor_for_batch(self, sentences):
        # names = self.stack_embedding.embeddings.get_names()
        lengths = [len(sentence)+2 for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        embedding_length=sentences[0][0].shape[0]
        # print(embedding_length,'asdfasdf')
        pre_allocated_zero_tensor = torch.zeros(
            embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=device,
        )
        sentence_list_to_tensor=[]
        for sentence in sentences:
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)-2

            t = pre_allocated_zero_tensor[: embedding_length * nb_padding_tokens]
            cls_embedding = torch.ones(embedding_length,device=device)*0.001
            # print(device,'????????',cls_embedding.device)
            sep_embedding=cls_embedding*2
            if t!=None:
                sentence_list_to_tensor.extend([cls_embedding,*sentence,sep_embedding,t])   #分隔符cls、sep和padding的embedding
            else:
                sentence_list_to_tensor.extend([cls_embedding, *sentence, sep_embedding])
        # for i in range(len(sentence_list_to_tensor)):
        # #     # sentence_list_to_tensor[i].to(device)
        #     print(sentence_list_to_tensor[i].device)
        sentence_tensor = torch.cat(sentence_list_to_tensor).contiguous().view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                embedding_length,
            ]
        )
        self.sequence_len=longest_token_sequence_in_batch
        return Variable(sentence_tensor)



        # for w,op in zip(weights,self._ops):
        #     op.embed(x)
        #     print(x.embedding.shape)
        #     try:
        #         self.embeding_1=torch.cat((self.embeding_1,w*x.embedding),dim=1)
        #     except AttributeError:
        #         print('???????????????????')
        #         self.embeding_1=w*x.embedding
        # a=torch.stack([w*op.embed(x) for w,op in zip(weights,self._ops)])
        # print(a.size(x),a)
        # return self.embeding_1
    # return sum(w * op(x) for w, op in zip(weights, self._ops))  #op(x)就是对输入x做一个相应的操作 w1*op1(x)+w2*op2(x)+...+w8*op8(x)
                                                                #也就是对输入x做8个操作并乘以相应的权重，把结果加起来

def _concat(xs):

  return torch.cat([x.view(-1).to(device) for x in xs])

class Architect(object):

  def __init__(self, model, config):
    self.config=config
    self.network_momentum = config.momentum
    self.network_weight_decay = config.weight_decay
    self.model=model
    self.optimizer = torch.optim.Adam([self.model.arch_parameters()],
        lr=config.arch_learning_rate, betas=(0.5, 0.999), weight_decay=config.arch_weight_decay)
  def _compute_unrolled_model(self, input, target,mask, eta, network_optimizer):
    model_feats=self.model(input)
    loss = self.model.loss(model_feats,mask, target)
    # self.model.to(device)
    # c=list(self.model.parameters())
    # for name,parameters in self.model.named_parameters():
        # print(name,parameters.device,parameters.size(),'>>>>>>>>')
    # print(self.model.arch_parameters(),'?????????')
    theta = _concat(filter(lambda p: p.requires_grad, self.model.parameters())).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.model.parameters()))).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train=None,target_train=None,mask_train=None,input_valid=None,masks_valid=None, target_valid=None,unrolled=None,eta=None,network_optimizer=None):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train,mask_train, input_valid, masks_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid,masks_valid, target_valid)
    nn.utils.clip_grad_norm_(self.model.arch_parameters(), 0.25)
    self.optimizer.step()

  def _backward_step(self, input_valid, masks_valid,target_valid):
    feats = self.model(input_valid)
    # tags,masks=target_valid,masks_valid
    #
    # # target_valid, masks_valid = Variable(target_valid).to(device), Variable(masks_valid).to(device)
    # for i, tag in enumerate(tags):
    #     tags[i] = torch.cat([tag, torch.zeros(self.model.batch_sequence_length - len(tag))])
    #     masks[i] = torch.cat([masks[i], torch.zeros(self.model.batch_sequence_length - len(tag))])
    # tags = Variable(torch.cat(tags).long()).view(self.config.batch_size, self.model.batch_sequence_length)
    # masks = Variable(torch.cat(masks).long()).view(self.config.batch_size, self.model.batch_sequence_length)
    # tags = tags.to(device)
    # masks = masks.to(device)

    loss = self.model.loss(feats, masks_valid, target_valid)
    l1_penalty=self.config.L1_weight*torch.sum(torch.abs(self.model.arch_parameters()-1))
    loss=loss+l1_penalty
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train,mask_train, input_valid, mask_valid,target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, mask_train,eta, network_optimizer)
    model_feat=unrolled_model(input_valid)
    unrolled_loss = unrolled_model.loss(model_feat,mask_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in [unrolled_model.arch_parameters()]]
    vector = [v.grad.data for v in filter(lambda x:x.requires_grad,unrolled_model.parameters())]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train,mask_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip([self.model.arch_parameters()], dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = copy.deepcopy(self.model)
    # model_new.backward()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      if v.requires_grad==True:
          v_length = np.prod(v.size())
          params[k] = theta[offset: offset+v_length].view(v.size())
          offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target,masks, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(filter(lambda x:x.requires_grad,self.model.parameters()), vector):
      p.data.add_(R, v)
    model_feats=self.model(input)
    loss = self.model.loss(model_feats,masks, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(filter(lambda x:x.requires_grad==True,self.model.parameters()), vector):
      p.data.sub_(2*R, v)
    model_feats = self.model(input)
    loss = self.model.loss(model_feats, masks, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(filter(lambda x:x.requires_grad==True,self.model.parameters()), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]




    # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    # n = input.size(0)
    # objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)

  #   if step % args.report_freq == 0:
  #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  #
  # return top1.avg, objs.avg

# c='i am a salt fish'
# sentence=Sentence(c)
# model=MixedOp()
# print(model(sentence,[0.4,0.2,0.2,0.5,0.3]))
