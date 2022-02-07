import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
import warnings
warnings.filterwarnings(action='ignore')


def get_label_metric_v4(hypothesis, reference):
    df_hyp = pd.read_csv(hypothesis, lineterminator='\n')
    df_ref = pd.read_csv(reference, lineterminator='\n')

    # positive label에 대해 계산
    df_hyp_pos1 = (df_hyp == 1).astype(int)
    del df_hyp_pos1["Reports"]
    df_hyp_pos1 = np.array(df_hyp_pos1)
    
    df_ref_pos1 = (df_ref == 1).astype(int)
    del df_ref_pos1["Reports"]
    df_ref_pos1 = np.array(df_ref_pos1)

    # zero에 대해 계산
    df_hyp_0 = (df_hyp == 0).astype(int)
    del df_hyp_0["Reports"]
    df_hyp_0 = np.array(df_hyp_0)

    df_ref_0 = (df_ref == 0).astype(int)
    del df_ref_0["Reports"]
    df_ref_0 = np.array(df_ref_0)

    # negative label에 대해 계산
    df_hyp_neg1 = (df_hyp == -1).astype(int)
    del df_hyp_neg1["Reports"]
    df_hyp_neg1 = np.array(df_hyp_neg1)

    df_ref_neg1 = (df_ref == -1).astype(int)
    del df_ref_neg1["Reports"]
    df_ref_neg1 = np.array(df_ref_neg1)

    # all label에 대해 acc 계산을 위한 것
    df_all_matching = (df_hyp.fillna(4) == df_ref.fillna(4)).astype(int)
    del df_all_matching["Reports"]
    df_all_matching = np.array(df_all_matching)

    ## all용 precision, recall, f1 계산을 위한 df_ref_all, df_hyp_all
    df_ref_all = df_ref_pos1 + df_ref_0 + df_ref_neg1
    df_hyp_all = df_hyp_pos1 + df_hyp_0 + df_hyp_neg1

    df_all_matching_exclude_TN = (df_hyp == df_ref).astype(int)
    del df_all_matching_exclude_TN["Reports"]
    df_all_matching_exclude_TN = np.array(df_all_matching_exclude_TN)



    # Accuarcy
    accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
    accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
    accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size
    accuracy_all = df_all_matching.sum() / df_all_matching.size

    # Precision
    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
    precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
    precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")
    precision_all = df_all_matching_exclude_TN.sum() / df_hyp_all.sum()

    # Recall
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
    recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
    recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")
    recall_all = df_all_matching_exclude_TN.sum() / df_ref_all.sum()

    # F1
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
    f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro")
    f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro")
    f1_all = 2 / (1/precision_all + 1/recall_all)

    # all에서 클래스별로 acc, precision, recall, f1구하기
    accuracy_all_list = []
    precision_all_list = []
    recall_all_list = []
    f1_all_list = []

    for i in range(df_all_matching.shape[1]):
        acc = df_all_matching[:,i].sum() / df_all_matching[:,i].size
        pcn = df_all_matching_exclude_TN[:,i].sum() / df_hyp_all[:,i].sum()
        rcl = df_all_matching_exclude_TN[:,i].sum() / df_ref_all[:,i].sum()
        f1 = 2 / (1/pcn + 1/rcl)
        accuracy_all_list.append(acc)
        precision_all_list.append(pcn)
        recall_all_list.append(rcl)
        f1_all_list.append(f1)

    return (accuracy_pos1, precision_pos1, recall_pos1, f1_pos1), (accuracy_0, precision_0, recall_0, f1_0), (accuracy_neg1, precision_neg1, recall_neg1, f1_neg1), (accuracy_all, precision_all, recall_all, f1_all), \
           accuracy_all_list, precision_all_list, recall_all_list, f1_all_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculating the metric')
    parser.add_argument('--hyp_file', type=str, default='/home/wcshin/scaleup_transformer/chexpert-labeler/labeled_GEN_epoch=000008_best_val_bleu_0.299_0.178_0.113_0.078_top_p_thres_0.9_tempr_0.7.csv',
                    help='chexpert labeler output csv file of GEN text')

    parser.add_argument('--ref_file', type=str, default='/home/wcshin/scaleup_transformer/chexpert-labeler/labeled_GT_epoch=000008_best_val_bleu_0.299_0.178_0.113_0.078_top_p_thres_0.9_tempr_0.7.csv',
                    help='chexpert labeler output csv file of GT text')

    args = parser.parse_args()

    metric_pos1, metric_0, metric_neg1, metric_all, accuracy_all_list, precision_all_list, recall_all_list, f1_all_list = \
        get_label_metric_v4(
            hypothesis = args.hyp_file, 
            reference = args.ref_file
            )


    print("(micro) accuracy, precision, recall, f1 for all : {}, {}, {}, {}".format(round(metric_all[0], 3), round(metric_all[1], 3), round(metric_all[2], 3), round(metric_all[3], 3)))


