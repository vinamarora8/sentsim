import torch
import torch.nn as nn

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

    def __init__(self, enc_device):
        super().__init__()

        # Load tokenizer and encoder
        from transformers import AutoTokenizer, AutoModel
        checkpoint = 'sentence-transformers/bert-base-nli-mean-tokens'
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_pretrained(checkpoint).to(enc_device)

        self.bias = nn.Parameter(torch.tensor(0.8))
        self.weight = nn.Parameter(torch.tensor(1.0))

        self.lin1 = nn.Linear(2 * 60 * 768, 768)
        self.lin2 = nn.Linear(2 * 60 * 768, 768)

        self.dropout = nn.Dropout(p=0.5)


    def forward(self, sent1, sent2):

        # Get embeddings from transformer encoder
        with torch.no_grad():
            tkn1 = self.tokenizer(sent1,
                                  padding='max_length',
                                  max_length=60,
                                  return_tensors='pt'
                                  )
            tkn2 = self.tokenizer(sent2,
                                  padding='max_length',
                                  max_length=60,
                                  return_tensors='pt'
                                  )

            tkn1 = tkn1.to(self.encoder.device)
            tkn2 = tkn2.to(self.encoder.device)

            emb1 = self.encoder(**tkn1)[0].detach().to(self.encoder.device)
            emb2 = self.encoder(**tkn2)[0].detach().to(self.encoder.device)

            att1 = tkn1['attention_mask'].to(self.encoder.device)
            att2 = tkn2['attention_mask'].to(self.encoder.device)

            x1 = apply_attention_mask(emb1, att1)
            x2 = apply_attention_mask(emb2, att2)


        # Conver to word x embedding 2D matrices into 1D vectors
        x1 = x1.flatten(-2)
        x2 = x2.flatten(-2)

        x = torch.cat((x1, x2), dim=-1)
        x = self.dropout(x)

        e1 = self.lin1(x)
        e2 = self.lin2(x)

        y = F.cosine_similarity(e1, e2).unsqueeze(-1)
        y = torch.sigmoid(self.weight*(y - self.bias))

        return y


model = SentSim_MeanPool(enc_device=device).to(device)

# Eval function


def eval_model(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    err = 0.0
    for i, data in enumerate(dataloader):

        model.eval()

        (sent1, sent2), y = data
        y = y.to(device)

        outputs = model.forward(sent1, sent2)
        outputs = (outputs > 0.5).float()

        err += (y - outputs).abs().mean() / len(dataloader)

    return err.item()


# Datasets

train_dataset = MsrPCDataset(split = 'train')
val_dataset = MsrPCDataset(split = 'val')


# Training

print(f'Initial train_data error {eval_model(train_dataset)}')
print()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.7)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(20):
    print(f'Epoch {epoch+1}: Val error: {eval_model(val_dataset)}')

    for i, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        (sent1, sent2), y = data
        outputs = model.forward(sent1, sent2)

        y = y.to(device).float()

        loss = criterion(outputs.reshape(-1), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Iteration {i}: Loss: {loss.item()}')

    lr_sched.step()
    print()

print(f'Final train_data_error: {eval_model(train_dataset)}')
print(f'Final val_data_error: {eval_model(val_dataset)}')
