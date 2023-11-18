from flair.embeddings import FlairEmbeddings,WordEmbeddings,PooledFlairEmbeddings,ELMoEmbeddings,TransformerWordEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

embedding_1=ELMoEmbeddings('small')

embedding_2=TransformerWordEmbeddings('bert-base-multilingual-cased')

embedding_3=WordEmbeddings('glove')
sentence=['sunday']
sentence=Sentence(sentence)
embedding_3.embed(sentence)
print(sentence.tokens[0])

embeddings=StackedEmbeddings([
        WordEmbeddings('glove'),
        ELMoEmbeddings('small'),
        FlairEmbeddings('mix-forward'),
        FlairEmbeddings('pubmed-forward'),

        PooledFlairEmbeddings('news-forward'),

        ])
text='i wish a champion'

sentence=['i', 'wish', 'a' ,'champion']
sentence=[sentence]
sentence.append(text.split(' '))
sentence=[Sentence(i) for i in sentence]
embeddings.embed(sentence)

embedding_type=[
        WordEmbeddings('glove'),

        FlairEmbeddings('mix-forward'),
        FlairEmbeddings('pubmed-forward'),
        PooledFlairEmbeddings('news-forward'),
        ]
print([i.embedding_length for i in embedding_type])

print(embeddings.embedding_length)
print(sentence[0].tokens[0].embedding.shape)