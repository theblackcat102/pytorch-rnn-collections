import torch
from torch import nn
from dataset import vocab_size, ImgCOCO, collate_fn
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.rnn = nn.LSTM(64, hidden_size, num_layers=num_layers, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, inputs):
        x = self.embedding(inputs)
        x, _ = self.rnn(x)
        return self.proj(x)




if __name__ == "__main__":
    dataset = ImgCOCO()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    for batch in train_dataloader:
        input_ids, label_ids = batch
        output = model(input_ids)
        loss = criterion(output.view(-1, vocab_size), label_ids.flatten())
        model.zero_grad()
        loss.backward()
        optimizer.step()



