import torch
import torch.nn.functional as F
from torch import nn
from dataset import vocab_size, ImgCOCO, collate_fn
from torch.utils.data import DataLoader


class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 vocab_size=vocab_size,
                 embd_size=64,
                 n_layers=2,
                 kernel=(5, 5),
                 out_chs=256,
                 res_block_count=1):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        # self.embd_size = embd_size

        self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(2, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.pool = nn.AvgPool2d((1, 60))
        self.fc = nn.Linear(out_chs, vocab_size)

    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        seq_len = x.size(1)
        x = self.embedding(x) # (bs, seq_len, embd_size)
        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * F.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * torch.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        # bs x cout x seq_len x 60 -> bs x seq_len x cout x 60
        h = h.permute(0, 2, 1, 3)
        h = self.pool(h).squeeze(-1)

        out = self.fc(h) # (bs, ans_size)

        return out

if __name__ == "__main__":
    dataset = ImgCOCO()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = GatedCNN(out_chs=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    for batch in train_dataloader:
        input_ids, label_ids = batch
        output = model(input_ids)
        loss = criterion(output.view(-1, vocab_size), label_ids.flatten())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())



