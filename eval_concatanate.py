import torch
from torch import nn
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
s = tokenizer(train_data.sentence[:N], padding=True, truncation=True, return_tensors='pt')
s = s.to(device)
match = torch.tensor(train_data.match[:N])

# Evaluate 
print('starting eval')

with torch.no_grad():
    model_output = model(**s)
    op = mean_pooling(model_output, s['attention_mask'])

# op = torch.clamp(op, min=0) 

sequence_shape = op.shape

class NeuralNetwork(nn.Module):
    def __init__(self, in_shape) -> None:
        super().__init__()
        # 3 layers, 100 each layer
        # need single binary number at the end
        self.in_size = in_shape
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_size, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


print('BEFORE TRAINING')
# for i in range(20):

train_steps = 1000
learning_rate = 1e-3

l2_norm = 0
nn_struct = NeuralNetwork(sequence_shape[1])
nn_loss = torch.nn.BCELoss()
print(nn_struct)
peek_accuracy_interval = 10
nn_struct = nn_struct.to(device)
nn_opt = torch.optim.Adam(nn_struct.parameters(),
                          learning_rate, weight_decay=l2_norm)
nn_opt.zero_grad()  # Setting our stored gradients equal to zero
nn_loss = nn_loss.to(device)
# nn_opt.to(device)
match = match.to(device)


error_threshold = 0.01


train_loader = torch.utils.data.DataLoader(op, batch_size=128, shuffle=True)

for i, data in enumerate(train_loader):
    outputs = nn_struct(data)
    loss = nn_loss(torch.squeeze(outputs), match.float())
    loss.backward()
    nn_opt.step()


    # peek into train/test every peek_accuracy_interval intervals
    if i % peek_accuracy_interval == 0:
        outputs = outputs.round().detach() 
        print(match)
        print(outputs)
        total_test = match.size(0)
        correct_test = torch.sum(outputs -
                                match.detach() > error_threshold)
        accuracy_test = 100 * correct_test/total_test
        print(
            f"Iteration: {i}. \nTest - Loss: {loss.item()}. Accuracy: {accuracy_test}")


err = 0.0
for i, data in enumerate(train_loader):

    nn_struct.eval()

    (sent1, sent2), y = data
    y = y.to(device)

    outputs = nn_struct(data)
    outputs = (outputs > 0.5).float()

    err += (y - outputs).abs().mean() / len(train_loader)

print(err.item())


