import pandas as pd 
from helpers import *
from clustering_helpers import *
from tslearn.metrics import cdist_dtw 
import warnings
warnings.filterwarnings("ignore")
RSEED = 2

Xtrain = pd.read_hdf('C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/data/train_test_set.h5', key='Xtrain') 
ytrain = pd.read_hdf('C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/data/train_test_set.h5', key='ytrain') 

list_of_features = list(Xtrain.drop(['hours_in', 'hours_in_', 'subject_id', 'hadm_id', 'icustay_id'], axis=1).columns)


Xtrain.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
Xtrain.head()

Xtrain = drop_recordings(Xtrain, 0.9)

def run_clustering(df, y_train, window):
    eval_metrics = pd.DataFrame(columns=["S_S", "PU"])
    labs_vitals_reduced = cut_time_series(df, window)
    labs_vitals_reduced
    df_reduced_pre_eucl, X_train,  hours_in_ = pre_eucl(labs_vitals_reduced, list_of_features)
    df_reduced_pre_eucl['hours_in'] = hours_in_
    pairwise_euclidean = calc_euclideans(df_reduced_pre_eucl, list_of_features)
    number_of_clusters = elbow_method(pairwise_euclidean,10)
    kmeans_labels, ward_labels, complete_labels, single_labels, labels_dict = run_algos(pairwise_euclidean, number_of_clusters)
    for keys, values in labels_dict.items():
        print("Clustering Algorithm: 8 windows", keys)
        s_s, pu = evaluation_metrics(pairwise_euclidean, values, y_train[y_train.index.isin(df_reduced_pre_eucl.reset_index().subject_id)].values, 'euclidean', rseed = RSEED)
        eval_metrics = eval_metrics.append({'S_S': s_s, 'PU': pu}, ignore_index=True)
        print("Silhouette Score: ", np.round(s_s, decimals=3) , "Purity: ", np.round(pu, decimals=3))
    plot_clusters(pairwise_euclidean, kmeans_labels, ward_labels, complete_labels, single_labels, y_train[y_train.index.isin(df_reduced_pre_eucl.reset_index().subject_id)].values, window)
    print("#######################################################################")
    print("############################ Next Window ##############################")

    return eval_metrics


eval_metrics = pd.DataFrame(columns=["S_S", "PU"])
window = [8, 16, 24, 32, 40, 48]
for i in window:
    eval_metrics_ = run_clustering(Xtrain, ytrain, i)
    eval_metrics = eval_metrics.append(eval_metrics_)
eval_metrics.to_csv("eval_metrics_baseline1.csv")

print("#######################################################################")
print("############################ DONE ##############################")