import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from msrpc_data_concatanate import MsrPCDataset

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

        self.lin1 = nn.Linear(120 * 768, 100)
        self.lin2 = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()
        
        # self.sig = nn.Linear(100, 1)

        self.tokenizer = tokenizer
        self.encoder = encoder


    def forward(self, sent):
        '''
        emb1: Embedding of first sentence
        att1: Attention mask for first sentence
        emb2: Embedding of second sentence
        att2: Attention mask for second sentence
        '''

        with torch.no_grad():
            tkn = self.tokenizer(sent, padding='max_length', max_length=120, return_tensors='pt')
            tkn = tkn.to(encoder.device)
            emb = self.encoder(**tkn)[0].detach().to(encoder.device)
            att = tkn['attention_mask'].to(encoder.device)
            x = apply_attention_mask(emb, att)

        '''
        x1 = torch.sum(x1, -2) # Sum over words
        x2 = torch.sum(x2, -2) # Sum over words
        '''

        x = x.flatten(-2)
        x = self.lin1(x)
        x = self.lin2(x)
        y = self.sig(x)
        return y



# Prepare dataset
checkpoint = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
checkpoint = 'sentence-transformers/bert-base-nli-mean-tokens'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
encoder = AutoModel.from_pretrained(checkpoint).to(device)

# dataset = MsrPCDataset(test=False)

train_dataset = MsrPCDataset(split = 'train')
val_dataset = MsrPCDataset(split = 'val')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

model = SentSim_MeanPool(tokenizer=tokenizer, encoder=encoder).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.7)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# model.train()

def eval_model(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    err = 0.0
    for i, data in enumerate(dataloader):

        model.eval()

        sent, y = data
        y = y.to(device)

        outputs = model.forward(sent)
        outputs = (outputs > 0.5).float()

        err += (y - outputs).abs().mean() / len(dataloader)

    return err

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.7)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
optimizer.zero_grad()
print(eval_model(train_dataset))



for epoch in range(20):
    print(f'Epoch {epoch}: Val error: {eval_model(val_dataset).item()}')

    for i, data in enumerate(train_loader):
        model.train()
        # optimizer.zero_grad()

        sent, y = data
        outputs = model.forward(sent)

        y = y.to(device).float()

        loss = criterion(outputs.reshape(-1), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Iteration {i}: Loss: {loss.item()}')
        if i%25 ==0:
            print(outputs.squeeze(), y.squeeze())

    lr_sched.step()

print(eval_model(train_dataset))