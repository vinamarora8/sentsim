import torch
from transformers import AutoTokenizer, AutoModel
from msrpc_data import MsrPCDataset
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
#train_data = MsrPCDataset(test=False, tokenizer=None)
train_data = MsrPCDataset(split = 'val')

N = len(train_data)

s1 = tokenizer(train_data.sentence1[:N], padding=True, truncation=True, return_tensors='pt')
s2 = tokenizer(train_data.sentence2[:N], padding=True, truncation=True, return_tensors='pt')

s1 = s1.to(device)
s2 = s2.to(device)

match = torch.tensor(train_data.match[:N])

# Evaluate
print('starting eval')

with torch.no_grad():
    model_output1 = model(**s1)
    model_output2 = model(**s2)

    op1 = mean_pooling(model_output1, s1['attention_mask'])
    op2 = mean_pooling(model_output2, s2['attention_mask'])

    sim = F.cosine_similarity(op1, op2).to('cpu')

sim = sim.to('cpu')
pred = (sim > 0.8).int()
err = (match - pred).abs().sum() / N
err = err.item()
print(err)

plt.scatter(match, sim)
plt.show()
