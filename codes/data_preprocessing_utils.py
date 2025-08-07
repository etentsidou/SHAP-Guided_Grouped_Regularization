import os, pickle
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from encoding import encode_by_crispr_net_method, encode_in_6_dimensions, encode_by_base_pair_vocabulary, encode_by_base_vocabulary

def get_epigenetic_code(epigenetic_1, epigenetic_2, epigenetic_3, epigenetic_4):
    epimap = {'A': 1, 'N': 0}
    tlen = 24
    epigenetic_1 = epigenetic_1.upper()
    epigenetic_1 = "N"*(tlen-len(epigenetic_1)) + epigenetic_1
    epigenetic_2 = epigenetic_2.upper()
    epigenetic_2 = "N"*(tlen-len(epigenetic_2)) + epigenetic_2
    epigenetic_3 = epigenetic_3.upper()
    epigenetic_3 = "N"*(tlen-len(epigenetic_3)) + epigenetic_3
    epigenetic_4 = epigenetic_4.upper()
    epigenetic_4 = "N"*(tlen-len(epigenetic_4)) + epigenetic_4
    epi_code = list()
    for i in range(len(epigenetic_1)):
        t = [epimap[epigenetic_1[i]], epimap[epigenetic_2[i]], epimap[epigenetic_3[i]], epimap[epigenetic_4[i]]]
        epi_code.append(t)
    return epi_code

def load_K562_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset K562 =====")
    k562_features = []
    k562_feature_ont = []
    k562_feature_offt = []
    k562_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/k562.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    # on_seqs = _ori_df[1].tolist()
    # off_seqs = _ori_df[6].tolist()
    # labels = _ori_df[11].tolist()
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        k562_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        k562_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        k562_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        k562_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
    k562_features = np.array(k562_features)
    k562_feature_ont = np.array(k562_feature_ont)
    k562_feature_offt = np.array(k562_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    if out_dim == 2:
        k562_labels = to_categorical(k562_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    k562_labels = np.array(k562_labels)
    print("[INFO] Encoded dataset K562 features with size of", k562_features.shape)
    print("[INFO] Encoded dataset K562 feature ont with size of", k562_feature_ont.shape)
    print("[INFO] Encoded dataset K562 feature offt with size of", k562_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset K562 is {0}, the active+inactive is {1}.".format(len(k562_labels[k562_labels>0]), len(k562_labels)))
    print("[INFO] Encoded dataset K562 on_epigenetic_code with size of", on_epigenetic_code.shape)
    print("[INFO] Encoded dataset K562 off_epigenetic_code with size of", off_epigenetic_code.shape)
    return k562_features, k562_feature_ont, k562_feature_offt, k562_labels, on_epigenetic_code, off_epigenetic_code

def load_HEK293t_encoded_by_both_base_and_base_pair(out_dim=1):
    print("[INFO] ===== Start Loading dataset HEK293t =====")
    hek293t_features = []
    hek293t_feature_ont = []
    hek293t_feature_offt = []
    hek293t_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/hek293t.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        hek293t_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        hek293t_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        hek293t_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        hek293t_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
    hek293t_features = np.array(hek293t_features)
    hek293t_feature_ont = np.array(hek293t_feature_ont)
    hek293t_feature_offt = np.array(hek293t_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    if out_dim == 2:
        hek293t_labels = to_categorical(hek293t_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    hek293t_labels = np.array(hek293t_labels)
    print("[INFO] Encoded dataset hek293t features with size of", hek293t_features.shape)
    print("[INFO] Encoded dataset hek293t feature ont with size of", hek293t_feature_ont.shape)
    print("[INFO] Encoded dataset hek293t feature offt with size of", hek293t_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset hek293t is {0}, the active+inactive is {1}.".format(len(hek293t_labels[hek293t_labels>0]), len(hek293t_labels)))
    print("[INFO] Encoded dataset HEK293t on_epigenetic_code with size of", on_epigenetic_code.shape)
    print("[INFO] Encoded dataset HEK293t off_epigenetic_code with size of", off_epigenetic_code.shape)
    return hek293t_features, hek293t_feature_ont, hek293t_feature_offt, hek293t_labels, on_epigenetic_code, off_epigenetic_code

# Grouped regularization functions (Epigenetics and Complete)

def load_K562_encoded_by_both_base_and_base_pair_with_group_regularization(out_dim=1):
    print("[INFO] ===== Start Loading dataset K562 =====")
    k562_features = []
    k562_feature_ont = []
    k562_feature_offt = []
    k562_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    k562_on_ctcf_features = []
    k562_off_ctcf_features = []
    k562_on_dnase_features = []
    k562_off_dnase_features = []
    k562_on_h3k4me3_features = []
    k562_off_h3k4me3_features = []
    k562_on_rrbs_features = []
    k562_off_rrbs_features = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/k562.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    # on_seqs = _ori_df[1].tolist()
    # off_seqs = _ori_df[6].tolist()
    # labels = _ori_df[11].tolist()
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        k562_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        k562_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        k562_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        k562_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))
    k562_features = np.array(k562_features)
    k562_feature_ont = np.array(k562_feature_ont)
    k562_feature_offt = np.array(k562_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    k562_on_ctcf_features = on_epigenetic_code[:, :, 0]  
    k562_off_ctcf_features = off_epigenetic_code[:, :, 0] 
    k562_on_dnase_features = on_epigenetic_code[:, :, 1]  
    k562_off_dnase_features = off_epigenetic_code[:, :, 1]
    k562_on_h3k4me3_features = on_epigenetic_code[:, :, 2]  
    k562_off_h3k4me3_features = off_epigenetic_code[:, :, 2]     
    k562_on_rrbs_features = on_epigenetic_code[:, :, 3]  
    k562_off_rrbs_features = off_epigenetic_code[:, :, 3]

    if out_dim == 2:
        k562_labels = to_categorical(k562_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    k562_labels = np.array(k562_labels)
    print("[INFO] Encoded dataset K562 features with size of", k562_features.shape)
    print("[INFO] Encoded dataset K562 feature ont with size of", k562_feature_ont.shape)
    print("[INFO] Encoded dataset K562 feature offt with size of", k562_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset K562 is {0}, the active+inactive is {1}.".format(len(k562_labels[k562_labels>0]), len(k562_labels)))
    # print("[INFO] Encoded dataset K562 on_epigenetic_code with size of", on_epigenetic_code.shape)
    # print("[INFO] Encoded dataset K562 off_epigenetic_code with size of", off_epigenetic_code.shape)
    print("[INFO] Encoded dataset K562 k562_on_ctcf_features with size of", k562_on_ctcf_features.shape)
    print("[INFO] Encoded dataset K562 k562_off_ctcf_features with size of", k562_off_ctcf_features.shape)
    print("[INFO] Encoded dataset K562 k562_on_dnase_features with size of", k562_on_dnase_features.shape)
    print("[INFO] Encoded dataset K562 k562_off_dnase_features with size of", k562_off_dnase_features.shape)
    print("[INFO] Encoded dataset K562 k562_on_h3k4me3_features with size of", k562_on_h3k4me3_features.shape)
    print("[INFO] Encoded dataset K562 k562_off_h3k4me3_features with size of", k562_off_h3k4me3_features.shape)
    print("[INFO] Encoded dataset K562 k562_on_rrbs_features with size of", k562_on_rrbs_features.shape)
    print("[INFO] Encoded dataset K562 k562_off_rrbs_features with size of", k562_off_rrbs_features.shape)
    return k562_features, k562_feature_ont, k562_feature_offt, k562_labels, k562_on_ctcf_features, k562_off_ctcf_features, k562_on_dnase_features, k562_off_dnase_features, k562_on_h3k4me3_features, k562_off_h3k4me3_features, k562_on_rrbs_features, k562_off_rrbs_features

def load_HEK293t_encoded_by_both_base_and_base_pair_with_group_regularization(out_dim=1):
    print("[INFO] ===== Start Loading dataset HEK293t =====")
    hek293t_features = []
    hek293t_feature_ont = []
    hek293t_feature_offt = []
    hek293t_labels = []
    on_epigenetic_code = []
    off_epigenetic_code = []
    hek293t_on_ctcf_features = []
    hek293t_off_ctcf_features = []
    hek293t_on_dnase_features = []
    hek293t_off_dnase_features = []
    hek293t_on_h3k4me3_features = []
    hek293t_off_h3k4me3_features = []
    hek293t_on_rrbs_features = []
    hek293t_off_rrbs_features = []
    ## load
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"/datasets/epigenetic_data/hek293t.epiotrt"
    _ori_df = pd.read_csv(data_path, sep='\t', index_col=None, header=None)
    _ori_df = shuffle(_ori_df, random_state=42)
    ## encode
    for i, row in _ori_df.iterrows():
        on_target_seq = row[1].upper()
        off_target_seq = row[6].upper()
        if "N" in off_target_seq:
            print(i, on_target_seq, off_target_seq)
            continue
        label = row[11]
        hek293t_features.append(encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq))
        hek293t_feature_ont.append(encode_by_base_vocabulary(seq=on_target_seq))
        hek293t_feature_offt.append(encode_by_base_vocabulary(seq=off_target_seq))
        hek293t_labels.append(label)
        on_epigenetic_code.append(get_epigenetic_code(row[2], row[3], row[4], row[5]))
        off_epigenetic_code.append(get_epigenetic_code(row[7], row[8], row[9], row[10]))

    hek293t_features = np.array(hek293t_features)
    hek293t_feature_ont = np.array(hek293t_feature_ont)
    hek293t_feature_offt = np.array(hek293t_feature_offt)
    on_epigenetic_code = np.array(on_epigenetic_code)
    off_epigenetic_code = np.array(off_epigenetic_code)
    hek293t_on_ctcf_features = on_epigenetic_code[:, :, 0]  
    hek293t_off_ctcf_features = off_epigenetic_code[:, :, 0] 
    hek293t_on_dnase_features = on_epigenetic_code[:, :, 1]  
    hek293t_off_dnase_features = off_epigenetic_code[:, :, 1]
    hek293t_on_h3k4me3_features = on_epigenetic_code[:, :, 2]  
    hek293t_off_h3k4me3_features = off_epigenetic_code[:, :, 2]     
    hek293t_on_rrbs_features = on_epigenetic_code[:, :, 3]  
    hek293t_off_rrbs_features = off_epigenetic_code[:, :, 3]

    if out_dim == 2:
        hek293t_labels = to_categorical(hek293t_labels) # 这个是自动one-hot化标签，0->[0. 1.] 1->[1. 0.]。二分类不需要这样，直接一位上用0和1就行。
    hek293t_labels = np.array(hek293t_labels)
    print("[INFO] Encoded dataset hek293t features with size of", hek293t_features.shape)
    print("[INFO] Encoded dataset hek293t feature ont with size of", hek293t_feature_ont.shape)
    print("[INFO] Encoded dataset hek293t feature offt with size of", hek293t_feature_offt.shape)
    print("[INFO] The labels number of active off-target sites in dataset hek293t is {0}, the active+inactive is {1}.".format(len(hek293t_labels[hek293t_labels>0]), len(hek293t_labels)))
    # print("[INFO] Encoded dataset HEK293t on_epigenetic_code with size of", on_epigenetic_code.shape)
    # print("[INFO] Encoded dataset HEK293t off_epigenetic_code with size of", off_epigenetic_code.shape)
    print("[INFO] Encoded dataset hek293t hek293t_on_ctcf_features with size of", hek293t_on_ctcf_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_off_ctcf_features with size of", hek293t_off_ctcf_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_on_dnase_features with size of", hek293t_on_dnase_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_off_dnase_features with size of", hek293t_off_dnase_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_on_h3k4me3_features with size of", hek293t_on_h3k4me3_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_off_h3k4me3_features with size of", hek293t_off_h3k4me3_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_on_rrbs_features with size of", hek293t_on_rrbs_features.shape)
    print("[INFO] Encoded dataset hek293t hek293t_off_rrbs_features with size of", hek293t_off_rrbs_features.shape)

    return hek293t_features, hek293t_feature_ont, hek293t_feature_offt, hek293t_labels, hek293t_on_ctcf_features, hek293t_off_ctcf_features, hek293t_on_dnase_features, hek293t_off_dnase_features, hek293t_on_h3k4me3_features, hek293t_off_h3k4me3_features, hek293t_on_rrbs_features, hek293t_off_rrbs_features


# if __name__=="__main__":
#     features, f2, f3, labels, on_epigenetic_code, off_epigenetic_code=load_HEK293t_encoded_by_both_base_and_base_pair()
#     print(labels.shape)
#     print(max(labels))
#     print(min(labels))
#     is_reg = False
#     for i in labels:
#         if 0<i<1:
#             is_reg = True
#             break
#     print(is_reg)


