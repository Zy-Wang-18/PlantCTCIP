import pandas as pd
import torch
import numpy as np
from torch.utils import data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class myDataset(data.Dataset):
    def __init__(self, e_list, p_list, label):
        self.e_list = e_list
        self.p_list = p_list
        self.label = label

    def __getitem__(self, index):
        gene_seq = self.e_list[index] + self.p_list[index]
        one_hot = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]}

        encode_list = []
        for element in gene_seq:
            encode_list.append(one_hot[element])
        seq = torch.tensor(encode_list, dtype=torch.float32).t().unsqueeze(dim=0)
        label = torch.tensor((self.label[index]), dtype=torch.int64)
        return seq, label

    def __len__(self):
        return len(self.e_list)


get_data = pd.read_csv('../final_dataset/ZS_RNAP2_final_dataset.csv')
e_list = get_data['e_seq']
p_list = get_data['p_seq']
label = get_data['label']

dataset = myDataset(e_list, p_list, label)

idx = np.arange(len(dataset))
slide1 = int(len(idx) * 0.8)
slide2 = int(len(idx) * 0.9)
train_idx = idx[:slide1]
val_idx = idx[slide1:slide2]
test_idx = idx[slide2:]

n_train = int(len(train_idx))
n_val = int(len(val_idx))
n_test = int(len(test_idx))
print('n_train, n_val, n_test', n_train, n_val, n_test)

trainset = data.Subset(dataset, train_idx)
valset = data.Subset(dataset, val_idx)
testset = data.Subset(dataset, test_idx)

train_loader = data.DataLoader(
    dataset=trainset,
    batch_size=32,
    shuffle=True
)

val_loader = data.DataLoader(
    dataset=valset,
    batch_size=32,
    shuffle=True
)

test_loader = data.DataLoader(
    dataset=testset,
    batch_size=32,
    shuffle=True
)
