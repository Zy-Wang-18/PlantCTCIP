import pandas as pd
import torch

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from torch.utils import data

class myDataset(data.Dataset):
    def __init__(self, d_name, d_list, p_name, p_list, label):
        self.d_name = d_name
        self.d_list = d_list
        self.p_name = p_name
        self.p_list = p_list
        self.label = label

    def __getitem__(self, index):
        gene_seq = self.d_list[index] + self.p_list[index]
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
        d_name = self.d_name[index]
        p_name = self.p_name[index]

        return d_name,p_name,seq, label

    def __len__(self):
        return len(self.d_list)

# file=['A2','B1','C1','D5','E1','F1','G1','K2']
# get_data = pd.read_csv('E:/a_Interaction/cotton/PDI/data/A2.csv')

get_data = pd.read_csv('../data/A2_double.csv',encoding='utf-8')
d_name = get_data['d_name']
d_list = get_data['d_list']
p_name = get_data['p_name']
p_list = get_data['p_list']
label = get_data['label']

dataset = myDataset(d_name, d_list, p_name, p_list, label)

train_ratio = 0.8
val_ratio = 0.1
n_train = int(len(dataset) * train_ratio)
n_val = int(len(dataset) * val_ratio)
n_test = len(dataset)-n_train-n_val
trainset0, valset0 ,testset0 = data.random_split(dataset, [n_train, n_val, n_test])


filtered_trainset = []
for d_name, p_name, seq, label in trainset0:
    # print(seq, trainset0[0], trainset0[0][0])
    # exit()
    '''tensor([[[0., 0., 1.,  ..., 1., 1., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         [1., 0., 0.,  ..., 0., 0., 0.]]])
('Chr10: 84759995-84760043', 'Garb_10G004730', tensor([[[0., 0., 1.,  ..., 1., 1., 0.],
         [0., 0., 0.,  ..., 0., 0., 1.],
         [0., 1., 0.,  ..., 0., 0., 0.],
         [1., 0., 0.,  ..., 0., 0., 0.]]]), tensor(0))
Chr10: 84759995-84760043'''

    if seq.shape[2] == trainset0[0][2].shape[2]:
        filtered_trainset.append((d_name,p_name,seq, label))
        # print(filtered_trainset)
        # exit()
        '''[(tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [1., 1., 0.,  ..., 1., 0., 0.],
         [0., 0., 1.,  ..., 0., 1., 1.],
         [0., 0., 0.,  ..., 0., 0., 0.]]]), tensor(1))]'''

filtered_valset = []
for d_name, p_name, seq, label in valset0:
    if seq.shape[2] == valset0[0][2].shape[2]:
        filtered_valset.append((d_name,p_name,seq, label))

filtered_testset = []
for d_name, p_name, seq, label in testset0:
    if seq.shape[2] == testset0[0][2].shape[2]:
        filtered_testset.append((d_name,p_name,seq, label))

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

d_name2 = []
d_list2 = []
p_name2 = []
p_list2 = []
label2 = []
for i in range(len(filtered_testset)):
    d_name_t = filtered_testset[i][0]
    p_name_t = filtered_testset[i][1]
    selected_rows = get_data.loc[(get_data['d_name'] == d_name_t) & (get_data['p_name'] == p_name_t), ['d_name', 'd_list', 'p_name', 'p_list','label']]
    # print(selected_rows)
    # exit()

    d_name2.append(selected_rows['d_name'].values[0])
    d_list2.append(selected_rows['d_list'].values[0])
    p_name2.append(selected_rows['p_name'].values[0])
    p_list2.append(selected_rows['p_list'].values[0])
    label2.append(selected_rows['label'].values[0])

df = pd.DataFrame({'d_name': d_name2, 'd_list': d_list2, 'p_name': p_name2, 'p_list': p_list2, 'label': label2})
df.to_csv('../data/test_data_A2.csv', index=False)

