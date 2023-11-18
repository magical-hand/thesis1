from torch.utils.data import Dataset
import torch
import random
from flair import device

class DataSetRewrite(Dataset):

    def __init__(self,data_path,max_length,label_dic):
        super(DataSetRewrite, self).__init__()

        self.data_path='./'+data_path

        file = open(self.data_path, encoding='utf-8')
        content = file.readlines()
        file.close()
        self.data_set=[]
        self.labels_set=[]
        self.masks_set=[]
        for line in content:
            text, label = line.strip().split('|||')
            tokens = text.split()
            label = label.split()
            if len(tokens) > max_length - 2:
                tokens = tokens[0:(max_length - 2)]
                label = label[0:(max_length - 2)]

            label_f = ["<start>"] + label + ['<eos>']
            # input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            tokens_f = ['<cls>'] + tokens + ['<sep>']
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * len(tokens_f)
            while len(label_ids) < max_length:
                tokens_f.append('<pad>')
                input_mask.append(0)
                label_ids.append(label_dic['<pad>'])
            # label_ids = torch.LongTensor(label_ids)
            # input_mask = torch.LongTensor(input_mask)
            assert len(tokens_f) == max_length
            assert len(input_mask) == max_length
            assert len(label_ids) == max_length
            self.data_set.append(' '.join(tokens_f))
            self.masks_set.append(input_mask)
            self.labels_set.append(label_ids)

        self.masks_set=torch.LongTensor(self.masks_set).to(device)
        self.labels_set = torch.LongTensor(self.labels_set).to(device)


    def __getitem__(self, index):
        if not isinstance(index, list):
            index=[index]
        return [self.data_set[i] for i in index],self.masks_set[index],self.labels_set[index]

    def __len__(self):
        return len(self.data_set)

class DataLoader_r():
    def __init__(self,dataset,batch_size,shuffle=True):
        self.batch_size=batch_size
        self.dataset=dataset
        self.shuffle=shuffle
        if self.shuffle==False:
            self.fetch_list=[range(0,len(self.dataset))]

    def __iter__(self):
        if self.shuffle==True:
            self.fetch_list=random.sample(range(0, len(self.dataset)), len(self.dataset))
        self.step=0
        return self

    def __next__(self):
        if self.step<len(self.dataset)//self.batch_size:
            sample_list=self.fetch_list[self.step*self.batch_size:(self.step+1)*self.batch_size]
            self.step += 1
            return self.dataset[sample_list]
        else:
            raise StopIteration
