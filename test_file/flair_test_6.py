# from flair.data import Sentence
#
#
# char=[['i',' have', 'a'],[ 'pan','right']]
#
# sentence=[]
# for x in char:
#     sentence.append(Sentence(x))
#
#
# # sentence=[Sentence(i) for i in char]
#
# print(sentence)
# -*- coding: utf-8 -*-

from flair.data import Sentence
# from flair.embeddings import WordEmbeddings

char='i have a pen'
sentence=Sentence(char)


print(sentence)
