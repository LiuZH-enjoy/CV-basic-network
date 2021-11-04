import torch
from model import Transformer
from tokenizer import Tokenizer

token = Tokenizer.from_word_json("word_index.json")
model = Transformer(7056, 7056, src_pad_idx=0, trg_pad_idx=0)
# model.load_state_dict(torch.load("transformer.pkl", map_location=torch.device("cpu")))

inp = "屋后千年树|"
inp = token.encoder_sentence([inp])
inp = torch.tensor(inp)[:, :-1]

output = torch.ones((1, 1), dtype=torch.int32)

for _ in range(10):
    pred_out = model(inp, output)
    pred_out = pred_out[:, -1:, :]
    pred_ids = torch.argmax(pred_out, dim=2)

    output = torch.cat([output, pred_ids], dim=1)
    if pred_ids == torch.tensor([[2]]):
        break

pred_out = torch.squeeze(output)
sentence = token.decoder_nums(pred_out.numpy())
print(sentence)

