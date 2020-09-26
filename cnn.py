import torch
import torch.nn.functional as F
from torch import nn
from dataset import vocab_size, ImgCOCO, collate_fn
from torch.utils.data import DataLoader
from absl import flags, app
from utils import set_seed, print_parameters
from tensorboardX import SummaryWriter
from tqdm import trange
import os

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_integer('total_epochs', 50, "total number of training steps")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_float('lr', 2e-4, "Generator learning rate")
flags.DEFINE_integer('num_layers', 2, "update Generator every this steps")
flags.DEFINE_integer('hidden_dim', 64, "hidden dimension")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_string('logdir', './logs/CNN_IMGCOCO', 'logging folder')

class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 vocab_size=vocab_size,
                 embd_size=64,
                 num_layers=2,
                 kernel=(5, 5),
                 hidden_size=256,
                 res_block_count=1):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        # self.embd_size = embd_size

        self.embedding = nn.Embedding(vocab_size, embd_size)
        out_chs = hidden_size
        n_layers = num_layers
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

def train():

    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))

    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    dataset = ImgCOCO()

    train_dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=collate_fn)

    model = GatedCNN(hidden_size=FLAGS.hidden_dim, num_layers=FLAGS.num_layers)
    print_parameters(model)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    step = 0
    for epoch in range(FLAGS.total_epochs):
        print('[Epoch %d]' % epoch)
        with trange(1, len(train_dataloader), dynamic_ncols=True) as pbar:
            for batch in train_dataloader:
                input_ids, label_ids = batch

                input_ids = input_ids.cuda()
                label_ids = label_ids.cuda()

                output = model(input_ids)
                loss = criterion(output.view(-1, vocab_size), label_ids.flatten())

                model.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalar("perplexity", torch.exp(loss).item(), step)

                step += 1

                pbar.set_postfix(loss="%.4f" % loss)
                pbar.update(1)

def main(argv):
    set_seed(FLAGS.seed)
    train()


if __name__ == '__main__':
    app.run(main)


