import torch
from transformers import AutoTokenizer, AutoModel
from msrpc_data_concatanate import MsrPCDataset
# from msrpc_data import MsrPCDataset
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set device
device = torch.device('cpu')
try:
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
except:
    pass

if (torch.cuda.is_available()):
    device = torch.device('cuda')

print(f"Device: {device}")


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/bert-base-nli-mean-tokens'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint).to(device)


# Load and tokenize data
N = 100
train_data = MsrPCDataset(test=False, tokenizer=None)
# print(train_data[0])
s = tokenizer(train_data.sentence[:N], padding=True, truncation=True, return_tensors='pt')
# print(s)
s = s.to(device)
match = torch.tensor(train_data.match[:N])

# Evaluate 
print('starting eval')

with torch.no_grad():
    model_output = model(**s)
    print(model_output)
    op = mean_pooling(model_output, s['attention_mask'])

    # print(op)
    # print(op.shape)

#     # sim = F.cosine_similarity(op1, op2).to('cpu')

pred = torch.sum(op, 1)
# print(len(pred))
# print(pred)
pred = pred.to('cpu')


# # sim = sim.to('cpu')
# # pred = (sim > 0.8).int()
# # err = (match - pred).abs().sum() / N
# # err = err.item()
# # print(err)

plt.scatter(match, pred)
plt.show()
