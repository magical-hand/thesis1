import flair

from flair.embeddings import FlairEmbeddings,WordEmbeddings,PooledFlairEmbeddings,ELMoEmbeddings,TransformerWordEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

# embedding_1=ELMoEmbeddings('small')
#
# embedding_2=TransformerWordEmbeddings('bert-base-multilingual-cased')
#
# embedding_3=WordEmbeddings('glove')
# sentence=['sunday']
# sentence=Sentence(sentence)
# embedding_3.embed(sentence)
# print(sentence.tokens[0])

embeddings=StackedEmbeddings([
        WordEmbeddings('glove'),
        ELMoEmbeddings('small'),
        FlairEmbeddings('mix-forward'),
        FlairEmbeddings('pubmed-forward'),

        PooledFlairEmbeddings('news-forward'),

        ])

sentence='i wish a champion'
sentence=Sentence(sentence)

embeddings.embed(sentence)

