import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

#          bos      eos       pad
id2token = ['<w/>','</w>', '<pad>','h', ';', '0', '7', '4', '1', '6', 'f', '3', 'w', '?', 'u', 'v', '.', 'm', '!', ' ', '8', '_', 's', '"', 'z', '$', 'y', 'c', ',', 'd', '9', 'a', '/', 'k', 'e', '2', '-', 'o', '&', 'i', 't', 'r', 'g', 'q', "'", 'l', 'x', ':', 'n', '#', 'j', 'b', '5', 'p']

token2id = { t: idx  for idx, t in enumerate(id2token) }

vocab_size = len(id2token)

class ImgCOCO(Dataset):


    def __init__(self, filename='image_coco.txt'):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                sent = line.strip()
                data.append({
                    'input_ids': ['<w/>']+list(sent),
                    'label_ids': list(sent)+['</w>']
                })
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    input_ids = []
    label_ids = []
    max_length = max([  len(s['input_ids']) for s  in batch ])
    for sent in batch:
        sent_tokens = []
        label_tokens = []
        for t in sent['input_ids']:
            sent_tokens.append(token2id[t])

        while len(sent_tokens) < max_length:
            sent_tokens.append(token2id['<pad>'])
        input_ids.append(sent_tokens)

        for t in sent['label_ids']:
            label_tokens.append(token2id[t])

        while len(label_tokens) < max_length:
            label_tokens.append(token2id['<pad>'])
        label_ids.append(label_tokens)

    label_ids = torch.from_numpy(np.array(label_ids)).long()
    input_ids = torch.from_numpy(np.array(input_ids)).long()
    return input_ids, label_ids

if __name__ == '__main__':

    dataset = ImgCOCO()

    dataloader = DataLoader(dataset,batch_size=32, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        input_ids, label_ids = batch
        print(input_ids.shape)