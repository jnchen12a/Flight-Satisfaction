import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(26)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(26, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class CustomDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = pd.read_csv(annotations_file).to_numpy().astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx, :-1], dtype=torch.float32), torch.tensor(self.data[idx, -1], dtype=torch.float32)
    
def preprocess(path, train=False):
    data = pd.read_csv(path)
    data = data.fillna(0)
    test = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])
    test['sat'] = pd.Categorical(test['satisfaction']).codes
    test = test.drop(columns=['num', 'id', 'satisfaction', 'Gender_Male'])
    if train:
        test.to_csv('./data/train_clean.csv', index=False)
    else:
        test.to_csv('./data/test_clean.csv', index=False)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.view(-1, 1)  # Reshape target to (64, 1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.view(-1, 1)  # Reshape target to (64, 1)
            pred = model(X)
            pred = (pred >= 0.5).float()
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(correct * 100):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    # preprocess the data
    preprocess('./data/test.csv')
    preprocess('./data/train.csv', train=True)

    # move data into dataloader
    testData = CustomDataset('./data/test_clean.csv')
    trainData = CustomDataset('./data/train_clean.csv')
    testDataloader = DataLoader(testData, batch_size=64, shuffle=True)
    trainDataloader = DataLoader(trainData, batch_size=64, shuffle=True)

    model = NeuralNetwork().to(device)

    lossFn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # run model
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainDataloader, model, lossFn, optimizer)
        test(testDataloader, model, lossFn)
    print("Done!")