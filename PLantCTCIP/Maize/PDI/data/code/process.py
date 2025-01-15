import numpy as np
import pandas as pd
from tqdm import tqdm
import Bio.SeqIO
import random
from random import shuffle
import pickle

def process(zu_name):
    df = pd.read_csv('../drop_repeat_dataset/' + zu_name + '_final_use.csv')
    ann1 = df['Annotation1'].values.astype('str').tolist()
    ann2 = df['Annotation2'].values.astype('str').tolist()

    geneID1, geneID2 = [], []
    seqs1, seqs2 = [], []
    for x in Bio.SeqIO.parse('../seqs/PDI_' + zu_name + '_use_seq.fasta', 'fasta'):
        if x.id[0:2] == 'Dm':
            geneID1.append(x.id)
            seqs1.append(str(x.seq[0:1500]))
        if x.id[0:2] == 'Zm':
            a = x.id.find('_')
            geneID1.append(x.id[:a])
            seqs1.append(str(x.seq[0:1500]))
    geneID1 = np.array([x.replace('"', '') for x in geneID1])
    seqs1 = np.array(seqs1)

    for x in Bio.SeqIO.parse('../seqs/PDI_' + zu_name + '_use_seq.fasta', 'fasta'):
        if x.id[0:2] == 'Dm':
            geneID2.append(x.id)
            seqs2.append(str(x.seq[0:1500]))
        if x.id[0:2] == 'Zm':
            a = x.id.find('_')
            geneID2.append(x.id[:a])
            seqs2.append(str(x.seq[0:1500]))
    geneID2 = np.array([x.replace('"', '') for x in geneID2])
    seqs2 = np.array(seqs2)

    j = 0
    seq1 = []
    seq2 = []
    temp_ann1 = []
    temp_ann2 = []
    temp_exp = []
    length = len(ann1)
    for i in range(length):
        tmp1 = seqs1[(geneID1 == ann1[i])]
        tmp2 = seqs2[(geneID2 == ann2[i])]
        if tmp1.size * tmp2.size > 0:
            if len(tmp1[0]) != 1500 or len(tmp2[0]) != 1500:
                print(len(tmp1[0]), ann1[i], len(tmp2[0]), ann2[i])
            if len(tmp1[0]) == 1500 and len(tmp2[0]) == 1500:
                seq1.append(tmp1[0])
                seq2.append(tmp2[0])
                temp_exp.append(1)
                temp_ann1.append(ann1[i])
                temp_ann2.append(ann2[i])
        if tmp1.size <= 0 or tmp2.size <= 0:
            j += 1
            print('Cuowu', tmp1.size, tmp2.size, ann1[i], ann2[i])
    print('j:', j)

    gene1 = []
    gene2 = []
    s1, s2 = [], []
    for x in Bio.SeqIO.parse('../seqs/PDI_' + zu_name + '_use_seq.fasta', 'fasta'):
        if x.id[0:2] == 'Zm':
            a = x.id.find('_')
            gene1.append(x.id[:a])
            s1.append(str(x.seq[0:1500]))
        if x.id[0:2] == 'Dm':
            gene2.append(x.id)
            s2.append(str(x.seq[0:1500]))
    gene1 = np.array([x.replace('"', '') for x in gene1])
    gene2 = np.array([x.replace('"', '') for x in gene2])
    s1 = np.array(s1)
    s2 = np.array(s2)

    pos_ann = []
    count = 0
    length = len(temp_ann1)
    print('length', len(temp_ann1))
    for i in range(len(temp_ann1)):
        pos_ann.append([temp_ann1[i], temp_ann2[i]])
    while count < length:
        a1 = random.choice(gene1)
        a2 = random.choice(gene2)
        if [a1, a2] not in pos_ann:
            tmp1 = s1[(gene1 == a1)]
            tmp2 = s2[(gene2 == a2)]
            if tmp1.size * tmp2.size > 0 and len(tmp1[0]) == 1500 and len(tmp2[0]) == 1500:
                count += 1
                seq1.append(tmp1[0])
                seq2.append(tmp2[0])
                temp_exp.append(0)
                temp_ann1.append(a1)
                temp_ann2.append(a2)
    print(len(temp_ann1), len(seq1), len(temp_ann2), len(seq2), len(temp_exp))

    df = pd.DataFrame(
        {'d_name': temp_ann1, 'd_seq': seq1, 'p_name': temp_ann2, 'p_seq': seq2, 'label': temp_exp})
    df.to_csv('../final_dataset/' + zu_name + '_PDI_merge.csv', index=False)

    # -------------------shuffle--------------------#
    temp = [temp_ann1, seq1, temp_ann2, seq2, temp_exp]
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
    
    df = pd.DataFrame({'d_name': e_name, 'd_seq': e_seq, 'p_name': p_name, 'p_seq': p_seq, 'label': label})
    df.to_csv('../final_dataset/' + zu_name + '_final_dataset.csv',index=False)


zu = ['ear', 'shoot', 'py']
for i in zu:
    process(i)
