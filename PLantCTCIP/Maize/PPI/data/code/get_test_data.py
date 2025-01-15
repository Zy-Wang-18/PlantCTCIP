import pandas as pd

def get_data(zu_name):
    get_data = pd.read_csv('../data/final_dataset/' + zu_name + '_final_dataset.csv')
    e_name = get_data['e_name']
    e_list = get_data['e_seq']
    p_list = get_data['p_seq']
    p_name = get_data['p_name']
    label = get_data['label']
    length = int(len(e_name)*0.9)

    new_e_name = e_name[length:]

    new_e_list = e_list[length:]
    new_p_name = p_name[length:]
    new_p_list = p_list[length:]
    new_lable = label[length:]
    df = pd.DataFrame(
        {'e_name': new_e_name, 'e_seq': new_e_list, 'p_name': new_p_name, 'p_seq': new_p_list, 'label': new_lable})
    df.to_csv('../test_data_' + zu_name + '_PPI.csv', index=False)


zu = ['ear', 'shoot', 'py', 'in_ear', 'in_tassel']
for i in zu:
    get_data(i)
