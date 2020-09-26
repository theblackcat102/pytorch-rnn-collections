import torch
from torch import nn
from dataset import vocab_size, ImgCOCO, collate_fn
from torch.utils.data import DataLoader
from absl import flags, app
from utils import set_seed
from utils import set_seed, print_parameters
from tensorboardX import SummaryWriter
from tqdm import trange
import os
from torchqrnn import QRNN

FLAGS = flags.FLAGS
# model and training
flags.DEFINE_integer('total_epochs', 50, "total number of training steps")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_float('lr', 2e-4, "Generator learning rate")
flags.DEFINE_integer('num_layers', 2, "update Generator every this steps")
flags.DEFINE_integer('hidden_dim', 512, "hidden dimension")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_string('logdir', './logs/QRNN_IMGCOCO', 'logging folder')


class QRNNModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2):
        super(QRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.rnn = QRNN(64, hidden_size, num_layers=num_layers)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, inputs):
        x = self.embedding(inputs)
        x = x.permute(1,0, 2)

        x, _ = self.rnn(x)
        x = x.permute(1,0, 2)

        return self.proj(x)

def train():

    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    writer = SummaryWriter(os.path.join(FLAGS.logdir))

    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    writer.add_text(
        "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    dataset = ImgCOCO()

    train_dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=collate_fn)

    model = QRNNModel(hidden_size=FLAGS.hidden_dim, num_layers=FLAGS.num_layers)
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

