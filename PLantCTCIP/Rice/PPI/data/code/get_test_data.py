import pandas as pd

def get_data(zu_name):
    get_data = pd.read_csv('/home/wzy/Interaction_Data_2023-10/rice/final_dataset/'+ zu_name +'_final_dataset.csv')
    # get_data=pd.read_csv('E:/nzl/cnn_trans/PPI/data/PPI(ear).csv')
    e_name = get_data['e_name']
    e_list = get_data['e_seq']
    p_list = get_data['p_seq']
    p_name = get_data['p_name']
    label = get_data['label']
    # print(len(d_name),len(d_list),len(p_list),len(p_name),len(label))
    length = int(len(e_name)*0.9)
    # print(length)
    new_e_name =e_name[length:]
    # print(new_d_name)
    new_e_list =e_list[length:]
    new_p_name =p_name[length:]
    new_p_list =p_list[length:]
    new_lable = label[length:]
    df = pd.DataFrame(
        {'e_name': new_e_name, 'e_seq': new_e_list, 'p_name': new_p_name, 'p_seq': new_p_list, 'label': new_lable})
    df.to_csv('/home/wzy/Interaction_Data_2023-10/rice/test_data/' + zu_name + '_test.csv', mode='w', header=True, index=None)

zu = ['ZS_H3K4','ZS_H3K9','ZS_RNAP2','MH_H3K4','MH_H3K9','MH_RNAP2']
for i in zu:
    get_data(i)