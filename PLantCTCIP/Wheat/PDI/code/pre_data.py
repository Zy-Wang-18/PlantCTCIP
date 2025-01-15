import pandas as pd
import torch
from torch.utils import data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
            if element in one_hot:
                encode_list.append(one_hot[element])
            else:
                encode_list.append([0, 0, 0, 0])
        seq = torch.tensor(encode_list, dtype=torch.float32).t().unsqueeze(dim=0)
        label = torch.tensor((self.label[index]), dtype=torch.int64)
        e_name = self.e_name[index]
        p_name = self.p_name[index]
        return e_name,p_name,seq, label

    def __len__(self):
        return len(self.e_list)

get_data = pd.read_csv('../data/wheat_double.csv',encoding = 'utf-8')
e_name = get_data['d_name']
e_list = get_data['d_list'].astype(str)
p_name = get_data['p_name']
p_list = get_data['p_list'].astype(str)
label = get_data['label']
dataset = myDataset(e_name, e_list, p_name, p_list, label)

train_ratio = 0.8
val_ratio = 0.1
n_train = int(len(dataset) * train_ratio)
n_val = int(len(dataset) * val_ratio)
n_test = len(dataset) - n_train - n_val

trainset, valset, testset = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])


filtered_trainset = []
for e_name, p_name, seq, label in trainset:
    if seq.shape[2] == trainset[0][2].shape[2]:
        filtered_trainset.append((e_name, p_name, seq, label))

filtered_valset = []
for e_name, p_name, seq, label in valset:
    if seq.shape[2] == valset[0][2].shape[2]:
        filtered_valset.append((e_name, p_name, seq, label))

filtered_testset = []
for e_name, p_name, seq, label in testset:
    if seq.shape[2] == testset[0][2].shape[2]:
        filtered_testset.append((e_name, p_name, seq, label))


train_loader = data.DataLoader(
    dataset=filtered_trainset,
    batch_size=32,
    shuffle=True
)

val_loader = data.DataLoader(
    dataset=filtered_valset,
    batch_size=32,
    shuffle=True
)

test_loader = data.DataLoader(
    dataset=filtered_testset,
    batch_size=32,
    shuffle=True
)


# save the test_data
e_name2 = []
e_list2 = []
p_name2 = []
p_list2 = []
label2 = []
for i in range(len(filtered_testset)):
    e_name_t = filtered_testset[i][0]
    p_name_t = filtered_testset[i][1]

    selected_rows = get_data.loc[(get_data['d_name'] == e_name_t) & (get_data['p_name'] == p_name_t),
    ['d_name', 'd_list', 'p_name', 'p_list','label']]

    e_name2.append(selected_rows['d_name'].values[0])
    e_list2.append(selected_rows['d_list'].values[0])
    p_name2.append(selected_rows['p_name'].values[0])
    p_list2.append(selected_rows['p_list'].values[0])
    label2.append(selected_rows['label'].values[0])

df = pd.DataFrame({'d_name': e_name2, 'd_list': e_list2, 'p_name': p_name2, 'p_list': p_list2, 'label': label2})
df.to_csv('../data/wheat_test.csv', index=False)
