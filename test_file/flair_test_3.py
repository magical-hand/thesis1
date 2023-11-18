import time
# print(time.time())

from flair.data import Sentence
# print(time.time())
from flair.embeddings import StackedEmbeddings
# print(time.time())
from embedding_generater import *
# print(time.time())
from config import Config

# embedding=TransformerWordEmbeddings('bert-base-uncased')
# print(time.time())
sentence='i have a pen'
sentence=Sentence(sentence)
# print(time.time())

# embedding.embed(sentence)
# print(sentence.tokens[0])
config=Config()
embeddings_list=embeddings_generater(config.dataset_config).embeddings_list
embedding=StackedEmbeddings(embeddings_list)
embedding.embed(sentence)
