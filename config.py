# coding=utf-8
from opration import embbedings_config

class Config(object):
    def __init__(self):
        data_path= 'data3/'
        self.label_file = data_path + "tag"
        self.train_file = data_path + "train_t1.txt"
        self.dev_file = data_path + "valid_t.txt"
        self.test_file = data_path + "test_t.txt"
        self.vocab = './data1/bert/vocab.txt'
        self.entity_list= data_path + 'entity_list'
        self.max_length = 100
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 16
        self.bert_path = 'data1/bert'
        self.rnn_hidden = 500
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 0.0005
        self.lr_decay = 0.001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 100
        self.momentum =(0.9, 0.999)
        self.network_weight_decay=0.001
        self.arch_learning_rate=0.0005
        self.arch_weight_decay=0.001
        self.L1_weight=0.1
        self.unrolled=False
        self.choosed=False
        self.embedding_select = [1,2,3,4,5,6,7,8,9,10]
        self.regular=False
        self.total_epoch=0
        self.dataset_config={
                'data_folder': r'C:\Users\Administrator\Desktop\ACE-main\datasets\conll_03_english',
                'column_format':{
                  0: 'text',
                  1: 'pos',
                  2: 'chunk',
                  3: 'ner'},
                # 'tag_to_bioes': 'ner'
        }

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':

    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
