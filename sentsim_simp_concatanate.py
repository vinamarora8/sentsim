import torch
import torch.nn as nn
import torch.nn.functional as F

class SentSim(nn.Module):

    def __init__(self, enc_device):
        super().__init__()

        # Load tokenizer and encoder
        from transformers import AutoTokenizer, AutoModel
        checkpoint = 'sentence-transformers/bert-base-nli-mean-tokens'
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_pretrained(checkpoint).to(enc_device)

        self.bias = nn.Parameter(torch.tensor(0.8))
        self.weight = nn.Parameter(torch.tensor(1.0))

        self.lin1 = nn.Linear(120 * 768, 100)
        self.lin2 = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()


        self.dropout = nn.Dropout(p=0.5)


    def forward(self, sentence):

        def apply_attention_mask(embeddings, mask):
            ''' Apply input token attention mask to output embeddings '''
            mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
            return embeddings * mask_expanded

        # Get embeddings from transformer encoder
        with torch.no_grad():
            tkn = self.tokenizer(sentence,
                                  padding='max_length',
                                  max_length=120,
                                  return_tensors='pt'
                                  )

            tkn = tkn.to(self.encoder.device)
            emb = self.encoder(**tkn)[0].detach().to(self.encoder.device)
            att = tkn['attention_mask'].to(self.encoder.device)
            x = apply_attention_mask(emb, att)


        # Conver to word x embedding 2D matrices into 1D vectors
        x = x.flatten(-2)
        # x = torch.cat((x1, x2), dim=-1)
        x = self.dropout(x)
        y = self.sig(self.lin2(self.lin1(x))) 
        # e = self.lin1(x)
        # y = F.cosine_similarity(e1, e2).unsqueeze(-1)
        # y = torch.sigmoid(self.weight*(y - self.bias))

        return y


def eval_model(model, dataset):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    err = 0.0
    for i, data in enumerate(dataloader):

        model.eval()

        (sent1, sent2), y = data
        y = y.to(device)
        
        # concatanate logic 
        sentence = []
        for i in range(len(sent1)):
            sentence.append((sent1[i], sent2[i]))

        # print(sentence)
        # sentence = sent1 + ' ' + sent2
        outputs = model.forward(sentence)
        outputs = (outputs > 0.5).float()

        err += (y - outputs).abs().mean() / len(dataloader)

    return err.item()


def train_model(model, train_dataset, val_dataset):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.7)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(30):
        print(f'Epoch {epoch+1}: Val error: {eval_model(model, val_dataset)}')

        for i, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            (sent1, sent2), y = data

            # concatanate logic 
            sentence = []
            for i in range(len(sent1)):
                sentence.append((sent1[i], sent2[i]))
            # sentence = sent1 + ' ' + sent2
            outputs = model.forward(sentence)

            y = y.to(device).float()

            loss = criterion(outputs.reshape(-1), y.reshape(-1))
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Iteration {i}: Loss: {loss.item()}')

        lr_sched.step()
        print()


def save_model(model, name):
    import os
    folder = "savestate"
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = folder + "/" + name + ".model"
    torch.save(model.state_dict(), filename)

    print(f"Model saved to {filename}")


if __name__ == '__main__':

    from msrpc_data import MsrPCDataset
    from time import gmtime, strftime

    train_dataset = MsrPCDataset(split = 'train')
    val_dataset = MsrPCDataset(split = 'val')

    device = torch.device('cuda')
    model = SentSim(enc_device=device).to(device)

    val_err = eval_model(model, val_dataset)
    print(f'Initial train_data_error: {eval_model(model, train_dataset)}')
    print(f'Initial val_data_error: {val_err}')
    print()

    train_model(model, train_dataset, val_dataset)

    val_err = eval_model(model, val_dataset)
    print(f'Final train_data_error: {eval_model(model, train_dataset)}')
    print(f'Final val_data_error: {val_err}')

    # Save model
    name = strftime("%m_%d_%H%M", gmtime()) + "_msrpc_val" + str(int(val_err * 100.0))
    save_model(model, name)
