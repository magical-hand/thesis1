from flair import embeddings as Embeddings
from opration import embbedings_config as embeddings
from config import Config


class embeddings_generater():
    def __init__(self,datasets_config):
        self.full_corpus = {'ner': 'CONLL_03_GERMAN:conll_03_new:CONLL_03_DUTCH:CONLL_03_SPANISH',
                        'upos': 'UD_GERMAN:UD_ENGLISH:UD_FRENCH:UD_ITALIAN:UD_DUTCH:UD_SPANISH:UD_PORTUGUESE:UD_CHINESE'}
        embedding_list=[]
        config=Config()
        for order_num,embedding in enumerate(embeddings):
            if order_num in config.embedding_select:
                if 'FastWordEmbeddings' in embedding:
                    embedding_list.append(
                        getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))
                elif 'FastCharacterEmbeddings' in embedding:
                    embedding_list.append(
                        getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))
                else:
                    embedding_list.append(getattr(Embeddings, embedding.split('-')[0])(**embeddings[embedding]))
        self.embeddings_list=embedding_list