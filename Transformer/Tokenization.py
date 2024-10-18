# input_ids
# input_type_ids
# attention_mask
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text_input = "Tôi đang học công nghệ thông tin"

token1 = tokenizer(text_input)
text1 = tokenizer.decode(token1['input_ids'])
print(text1)


token2 = tokenizer.encode(text_input)
text2 = tokenizer.decode(token2)
print(text2)