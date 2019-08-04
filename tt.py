from pytorch_transformers.tokenization_roberta import (Dictionary,
                                                       RobertaTokenizer)

TEXT = 'Hello world! cécé herlolip'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

print(
    tokenizer.encode(TEXT)
)
token_ids = [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]
# print(tokenizer.convert_ids_to_tokens(token_ids))
print(list ( tokenizer._convert_id_to_token(t) for t in token_ids ))
print(tokenizer.decode(token_ids))
# print(tokenizer._convert_id_to_token(31414))

# dictionary = Dictionary.load('./dict.txt')

# print(
# 	dictionary.encode_line('<s> 15496 995 0 269 2634 32682 607 47288 541 </s>', append_eos=False)
# )


print("khlj")