import pandas as pd

def get_data(zu_name):
    get_data = pd.read_csv('../final_dataset/' + zu_name + '_final_dataset.csv')
    d_name = get_data['d_name']
    d_list = get_data['d_seq']
    p_list = get_data['p_seq']
    p_name = get_data['p_name']
    label = get_data['label']

    length = int(len(d_name)*0.9)
    new_d_name = d_name[length:]
    new_d_list = d_list[length:]
    new_p_name = p_name[length:]
    new_p_list = p_list[length:]
    new_lable = label[length:]
    df = pd.DataFrame(
        {'d_name': new_d_name, 'd_seq': new_d_list,
         'p_name': new_p_name, 'p_seq': new_p_list,
         'label': new_lable})
    df.to_csv('../test_data_' + zu_name + '_PDI.csv', index=False)


zu = ['ear', 'shoot', 'py']
for i in zu:
    get_data(i)
