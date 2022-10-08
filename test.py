# -*- coding: utf-8 -*-
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
#print(encoded_input)
#print(tokenizer.convert_ids_to_tokens(encoded_input['input_ids'].squeeze()))
output = model(**encoded_input)

#print(model)
print(output.pooler_output.shape)
