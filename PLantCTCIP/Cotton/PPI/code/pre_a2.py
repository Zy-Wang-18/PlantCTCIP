import pandas as pd
import torch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from torch.utils import data

class myDataset(data.Dataset):
    def __init__(self, e_name, e_list, p_name, p_list, label):
        self.e_name = e_name
        self.e_list = e_list
        self.p_name = p_name
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
        e_name = self.e_name[index]
        p_name = self.p_name[index]

        return e_name, p_name, seq, label

    def __len__(self):
        return len(self.e_list)

# train_data
get_data = pd.read_csv('../data/train_data_A2.csv')
e_name0 = get_data['e_name']
e_list0 = get_data['e_list']
p_name0 = get_data['p_name']
p_list0 = get_data['p_list']
label0 = get_data['label']
trainset = myDataset(e_name0, e_list0, p_name0, p_list0, label0)
n_train = len(trainset)
print('n_train:', n_train)

#val_data
get_data = pd.read_csv('../data/val_data_A2.csv')
e_name1 = get_data['e_name']
e_list1 = get_data['e_list']
p_name1 = get_data['p_name']
p_list1 = get_data['p_list']
label1 = get_data['label']
valset = myDataset(e_name1, e_list1, p_name1, p_list1, label1)
n_val = len(valset)
print('n_val:', n_val)

#test_data
get_data = pd.read_csv('../data/test_data_A2.csv')
e_name2 = get_data['e_name']
e_list2 = get_data['e_list']
p_name2 = get_data['p_name']
p_list2 = get_data['p_list']
label2 = get_data['label']
testset = myDataset(e_name2, e_list2, p_name2, p_list2, label2)
n_test = len(testset)
print('n_test:', n_test)

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