import torch
import torch.nn as nn

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


def apply_attention_mask(embeddings, mask):
    ''' Apply input token attention mask to output embeddings '''
    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return embeddings * mask_expanded

class SentSim_MeanPool(nn.Module):

    def __init__(self, tokenizer, encoder):
        super().__init__()

        self.bias = nn.Parameter(torch.tensor(0.8))
        self.weight = nn.Parameter(torch.tensor(1.0))

        self.tokenizer = tokenizer
        self.encoder = encoder


    def forward(self, sent1, sent2):
        '''
        emb1: Embedding of first sentence
        att1: Attention mask for first sentence

        emb2: Embedding of second sentence
        att2: Attention mask for second sentence
        '''

        with torch.no_grad():
            tkn1 = self.tokenizer(sent1, padding='max_length', max_length=60, return_tensors='pt')
            tkn2 = self.tokenizer(sent2, padding='max_length', max_length=60, return_tensors='pt')

            tkn1 = tkn1.to(encoder.device)
            tkn2 = tkn2.to(encoder.device)

            emb1 = self.encoder(**tkn1)[0].detach().to(encoder.device)
            emb2 = self.encoder(**tkn2)[0].detach().to(encoder.device)

            att1 = tkn1['attention_mask'].to(encoder.device)
            att2 = tkn2['attention_mask'].to(encoder.device)

            x1 = apply_attention_mask(emb1, att1)
            x2 = apply_attention_mask(emb2, att2)

        x1 = torch.sum(x1, -2) # Sum over words
        x2 = torch.sum(x2, -2) # Sum over words

        y = F.cosine_similarity(x1, x2).unsqueeze(-1)

        #y = self.lin_op(y)
        y = torch.sigmoid(self.weight*(y - self.bias))

        return y



# Prepare dataset
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/bert-base-nli-mean-tokens'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
encoder = AutoModel.from_pretrained(checkpoint).to(device)

dataset = MsrPCDataset(test=False)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

model = SentSim_MeanPool(tokenizer=tokenizer, encoder=encoder).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.7)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model.train()
for epoch in range(50):
    print(f'EPOCH {epoch}')


    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        (sent1, sent2), y = data
        outputs = model.forward(sent1, sent2)

        y = y.to(device).float()

        loss = criterion(outputs.reshape(-1), y.reshape(-1))
        loss.backward()
        optimizer.step()

    print(loss.item(), model.bias.item(), model.weight.item())
    lr_sched.step()


err = 0.0
for i, data in enumerate(train_loader):

    model.eval()

    (sent1, sent2), y = data
    y = y.to(device)

    outputs = model.forward(sent1, sent2)
    outputs = (outputs > 0.5).float()

    err += (y - outputs).abs().mean() / len(train_loader)

print(err.item())
