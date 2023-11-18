import random

def div_data(file_path,output_path,div_part,data_type):
    data_length=0
    with open(file_path,encoding='utf-8') as f1:
        if data_type==1:
            for line in f1:
                if line=='\n':
                    data_length+=1
        else:
            for _ in f1:
                data_length+=1
    with open(output_path+str(div_part)+str(data_type),'w',encoding='utf-8') as w:
        with open(file_path,encoding='UTF-8') as f:
            rand_list = random.sample(range(0, data_length), data_length // div_part)
            step=0
            for line in f:
                if step in rand_list:
                    w.write(line)
                if data_type==1:
                    if line=='\n':
                        step+=1
                else:
                    step+=1

if __name__=='__main__':
    random.seed(10)
    for div_part in [3,9]:
        div_data('../conll_03_english_tot/train.txt', './output_data', div_part, 1)