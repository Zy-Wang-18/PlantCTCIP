import numpy as np
import pandas as pd
from tqdm import tqdm

from random import shuffle
import pickle

def process(zu_name):
    df = pd.read_csv('../' + zu_name + '.csv')
    ann1 = df['e_name'].values.astype('str').tolist()
    ann2 = df['p_name'].values.astype('str').tolist()
    seq1 = df['e_seq'].values.astype('str').tolist()
    seq2 = df['p_seq'].values.astype('str').tolist()
    label = df['label'].values.astype('str').tolist()
    print(len(ann1), len(seq1), len(ann2), len(seq2), len(label))
    i = len(ann1)

    ann1_double = ann2
    ann2_double = ann1
    seq1_double = seq2
    seq2_double = seq1
    ann1.extend(ann1_double)
    ann2.extend(ann2_double[:i])
    seq1.extend(seq1_double)
    seq2.extend(seq2_double[:i])
    label.extend(label)
    print(len(ann1), len(seq1), len(ann2), len(seq2), len(label))
    df = pd.DataFrame(
        {'e_name': ann1, 'e_seq': seq1, 'p_name': ann2, 'p_seq': seq2, 'label': label})
    df.to_csv('../final_dataset/' + zu_name + '_merge.csv', index=False)

    # -------------------shuffle--------------------#
    temp = [ann1, seq1, ann2, seq2, label]
    temp = list(zip(*temp))
    shuffle(temp)
    
    e_name = [i[0] for i in temp]
    e_seq = [i[1] for i in temp]
    p_name = [i[2] for i in temp]
    p_seq = [i[3] for i in temp]
    label = [i[4] for i in temp]
    
    # ----------------encoding-----------------#
    d = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
         'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1],
         'N': [0, 0, 0, 0]}
    
    def encode(seq):
        tmp = []
        for i in seq:
            tmp.append(d[i])
        return np.array(tmp, dtype='float16')
    
    dna1 = []
    dna2 = []
    for i in tqdm(range(len(e_name))):
        dna1.append(encode(e_seq[i]))
        dna2.append(encode(p_seq[i]))
    print(len(dna1))
    print(len(dna2))
    dna1 = np.array(dna1, dtype='float16')
    dna2 = np.array(dna2, dtype='float16')
    
    # --------------------save data----------------#
    pickle.dump(dna1, open('../final_dataset/' + zu_name + '_dna1', 'wb'), protocol=4)
    pickle.dump(dna2, open('../final_dataset/' + zu_name + '_dna2', 'wb'), protocol=4)
    
    df = pd.DataFrame({'e_name': e_name, 'e_seq': e_seq, 'p_name': p_name, 'p_seq': p_seq, 'label': label})
    df.to_csv('../final_dataset/' + zu_name + '_final_dataset.csv', index=False)


zu = ['ZS_H3K4', 'ZS_H3K9', 'ZS_RNAP2', 'MH_H3K4', 'MH_H3K9', 'MH_RNAP2']
for i in zu:
    process(i)
