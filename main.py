import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "This is customer support for Freshly."
text_2 = "Thanks for contacting "
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)

# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])


# # Load pre-trained model (weights)
# model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
# model.eval()
#
# # If you have a GPU, put everything on cuda
# # tokens_tensor_1 = tokens_tensor_1.to('cuda')
# # tokens_tensor_2 = tokens_tensor_2.to('cuda')
# # model.to('cuda')
#
# with torch.no_grad():
#     # Predict hidden states features for each layer
#     hidden_states_1, mems_1 = model(tokens_tensor_1)
#     # We can re-use the memory cells in a subsequent call to attend a longer context
#     hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# Load pre-trained model (weights)
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()

# If you have a GPU, put everything on cuda
#tokens_tensor_1 = tokens_tensor_1.to('cuda')
#tokens_tensor_2 = tokens_tensor_2.to('cuda')
#model.to('cuda')

with torch.no_grad():
    # Predict all tokens
    predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)


# get the predicted last token
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

#assert predicted_token == 'who'

text_predicted = [predicted_token]

for i in range(10):

    predictions_2, mems_2 = model(predicted_index * torch.ones([1, 1], dtype=tokens_tensor_2.dtype), mems=mems_2)

    predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    text_predicted += [predicted_token]


print(' '.join(text_predicted))