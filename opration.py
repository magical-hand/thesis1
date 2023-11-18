from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
stacked_embeddings = StackedEmbeddings([
                                        WordEmbeddings('glove'),
                                        FlairEmbeddings('news-forward'),
                                        FlairEmbeddings('news-backward'),
                                       ])


# PRIMITIVES= [
#     'glove',
#     # 'small',
#     'mix-forward',
#     'pubmed-forward',
#     # 'en-impresso-hipe-v1-forward'
# ]

OPS = [
    # "BytePairEmbeddings('en')",
    "WordEmbeddings('glove')",
    "TransformerWordEmbeddings('bert-base-uncased')",
    "FlairEmbeddings('mix-forward')",
    "FlairEmbeddings('pubmed-forward')",
    "TransformerWordEmbeddings('bert-base-multilingual-cased')",
    # "PooledFlairEmbeddings('news-forward')",

    "CharacterEmbeddings()",
    "ELMoEmbeddings('small')"
]

embbedings_config={'TransformerWordEmbeddings-1': {'model': 'bert-base-cased', 'layers': '-1,-2,-3,-4',
                                         'pooling_operation': 'mean',
                                         'embedding_name': '/home/yongjiang.jy/.cache/torch/transformers/bert-base-cased'},
         'TransformerWordEmbeddings-2': {'model': 'roberta-base', 'layers': '-1,-2,-3,-4',
                                         'pooling_operation': 'mean'},
         'ELMoEmbeddings-0': {'model': 'original','options_file':r'C:\Users\Administrator\Desktop\keykeykey - 简单一阶近似\model_ELMO\options.json','weight_file':r'C:\Users\Administrator\Desktop\keykeykey - 简单一阶近似\model_ELMO\elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'},
         'CharacterEmbeddings': {'char_embedding_dim': 25, 'hidden_size_char': 25},
         'WordEmbeddings-0': {'embeddings': 'glove'},
         'WordEmbeddings-1': {'embeddings': 'en'}, 'FlairEmbeddings-0': {'model': 'en-forward'},
         'FlairEmbeddings-1': {'model': 'en-backward'}, 'FlairEmbeddings-2': {'model': 'multi-forward'},
         'FlairEmbeddings-3': {'model': 'multi-backward'},
         'TransformerWordEmbeddings-0': {'layers': '-1', 'pooling_operation': 'first',
                                         'model': 'xlm-roberta-large-finetuned-conll03-english',
                                         'embedding_name': '/home/yongjiang.jy/.flair/embeddings/xlm-roberta-large-finetuned-conll03-english'}}
    
    
    
    
    
    
#   'none' : lambda C, stride, affine: Zero(stride),
#   'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
#   'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
#   'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#   'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
#   'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
#   'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#   'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
#   'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
#   'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
#     nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#     ),
# }