from pytorch_transformers.modeling_bert import BertConfig, BertModel, BertEmbeddings
import torch
import numpy as np


TEXT = 'Hello world! cécé herlolip'


config = BertConfig(
    vocab_size_or_config_json_file=50265,
    hidden_size=768,
    num_hidden_layers=12,
    max_position_embeddings=514,
    type_vocab_size=1,
)

print(config)

model = BertModel(config)
model.eval()

input_ids = torch.tensor([    0, 31414,   232,   328,   740,  1140, 12695,    69, 46078,  1588,   2]).unsqueeze(0)  # Batch size 1
print(input_ids)
# outputs = model(input_ids)




###################
## stick roberta's weights in there.
#####
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/Users/gibbon/Desktop/fairseq/roberta.base/')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
roberta_sent_encoder = roberta.model.decoder.sentence_encoder

#####
embeddings: BertEmbeddings = model.embeddings

embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
embeddings.token_type_embeddings.weight.data = torch.zeros_like(embeddings.token_type_embeddings.weight)
embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias
embeddings.LayerNorm.variance_epsilon = roberta_sent_encoder.emb_layer_norm.eps

####
# seq_length = input_ids.size(1)
# position_ids = torch.arange(2, seq_length+2, dtype=torch.long, device=input_ids.device)
# position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
# token_type_ids = torch.zeros_like(input_ids)

# x = embeddings.word_embeddings(input_ids) + embeddings.position_embeddings(position_ids) + embeddings.token_type_embeddings(token_type_ids)

print(
    embeddings(input_ids)
)


"42"