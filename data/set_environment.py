# set environment

!pip install effdet
!pip install timm
!pip install -U albumentations
!pip install pytorchcv
!pip install transformers

import sys
sys.path.insert(0, "timm-efficientdet-pytorch")
sys.path.insert(0, "omegaconf")

special_tokens = '[pad]', '[start]', '[end]'

from transformers import BertTokenizer, BertConfig, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_config = BertConfig.from_pretrained('bert-base-uncased')
bert_model = BertModel(bert_config).from_pretrained('bert-base-uncased', config = bert_config)
bert_tokenizer.add_tokens(special_tokens)
bert_model.resize_token_embeddings(len(bert_tokenizer)) 


# colab
from google.colab import drive
drive.mount('/content/gdrive/', force_remount = True)
