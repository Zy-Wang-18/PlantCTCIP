import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# file = ['A2', 'B1', 'C1', 'D5', 'E1', 'F1', 'G1', 'K2']
file = ['A2']
for k in tqdm(range(len(file))):
    data = pd.read_csv('../data/' + file[k] + '_double.csv', encoding='utf-8')

    train_data, remaining_data = train_test_split(data, train_size=0.8, random_state=42)
    val_data, test_data = train_test_split(remaining_data, train_size=0.5, random_state=42)

    train_data.to_csv('../data/train_data_' + file[k] + '.csv', index=False)
    val_data.to_csv('../data/val_data_' + file[k] + '.csv', index=False)
    test_data.to_csv('../data/test_data_' + file[k] + '.csv', index=False)
