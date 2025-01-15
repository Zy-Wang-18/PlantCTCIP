import pandas as pd
import Bio.SeqIO

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dict_seq_HC_tss = {}
dict_seq_HC_tts = {}

data = pd.read_csv('../data/wheat_positive_name_double_drop_repeat.csv', encoding='utf-8')
e_name = data['e_name'].values.astype('str').tolist()
p_name = data['p_name'].values.astype('str').tolist()

len_sample = len(data)


for seq in Bio.SeqIO.parse('../data/original_data/Wheat_HC_gene_tss_1.5k_seq.fasta', 'fasta'):
    dict_seq_HC_tss[seq.id[0:18]] = str(seq.seq)
for seq in Bio.SeqIO.parse('../data/original_data/Wheat_HC_gene_tts_1.5k_seq.fasta', 'fasta'):
    dict_seq_HC_tts[seq.id[0:18]] = str(seq.seq)


data_n = pd.read_csv('../data/wheat_negative_name_double_drop_repeat.csv', encoding='utf-8')

en_name = data_n['e_name'].values.astype('str').tolist()
pn_name = data_n['p_name'].values.astype('str').tolist()

e_list = []
e_name1 = []
p_list = []
p_name1 = []
label = []

print(len(e_name), len(en_name))
length = min(len(e_name), len(en_name))
for i in range(length):
    name_e = e_name[i]
    name_p = p_name[i]

    name_en = en_name[i]
    name_pn = pn_name[i]

    e_seq1 = dict_seq_HC_tss.get(name_e, None)
    e_seq2 = dict_seq_HC_tts.get(name_e, None)
    e_seq = f"{e_seq1}{e_seq2}"

    p_seq1 = dict_seq_HC_tss.get(name_p, None)
    p_seq2 = dict_seq_HC_tts.get(name_p, None)
    p_seq = f"{p_seq1}{p_seq2}"

    en_seq1 = dict_seq_HC_tss.get(name_en, None)
    en_seq2 = dict_seq_HC_tts.get(name_en, None)
    en_seq = f"{en_seq1}{en_seq2}"

    pn_seq1 = dict_seq_HC_tss.get(name_pn, None)
    pn_seq2 = dict_seq_HC_tts.get(name_pn, None)
    pn_seq = f"{pn_seq1}{pn_seq2}"

    missing_sequence = False
    if e_seq1 is None or e_seq2 is None or p_seq1 is None or p_seq2 is None or en_seq1 is None or en_seq2 is None or pn_seq1 is None or pn_seq2 is None:
        missing_sequence = True

    if not missing_sequence:
        e_name1.append(name_e)
        e_list.append(e_seq)
        p_name1.append(name_p)
        p_list.append(p_seq)
        label.append(1)

        e_name1.append(name_en)
        e_list.append(en_seq)
        p_name1.append(name_pn)
        p_list.append(pn_seq)
        label.append(0)

df = pd.DataFrame({'e_name': e_name1, 'e_list': e_list, 'p_name': p_name1, 'p_list': p_list, 'label': label})
df.to_csv('../data/wheat_name_list.csv', index=False)