import pandas as pd 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.cluster import KMeans, AgglomerativeClustering
import helper_functions.helpers as helpers
import helper_functions.clustering_helpers as clustering_helpers
import os 
print(os.getcwd())

import helper_functions.helpers as helpers
import warnings
warnings.filterwarnings("ignore")


RSEED = 2


def run_algos(dtw_dist, number_of_clusters):
    labels_dict = {}
    
    kmeans = KMeans(n_clusters = number_of_clusters, init = 'k-means++',  random_state = RSEED)
    kmeans_labels = kmeans.fit_predict(dtw_dist)
    labels_dict["kmeans"] = kmeans_labels

    ward = AgglomerativeClustering(n_clusters=number_of_clusters)
    ward_labels = ward.fit_predict(dtw_dist)
    labels_dict["ward"] = ward_labels

    complete = AgglomerativeClustering(n_clusters=number_of_clusters, linkage='complete')
    complete_labels  = complete.fit_predict(dtw_dist)
    labels_dict["complete"] = complete_labels

    single = AgglomerativeClustering(n_clusters=number_of_clusters, linkage='single')
    single_labels  = single.fit_predict(dtw_dist)
    labels_dict["single"] = single_labels

    return kmeans_labels, ward_labels, complete_labels, single_labels, labels_dict


Xtrain = pd.read_hdf('C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/train_test_set.h5', key='Xtrain') 
ytrain = pd.read_hdf('C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/train_test_set.h5', key='ytrain') 

list_of_features = list(Xtrain.drop(['hours_in', 'subject_id', 'hadm_id', 'icustay_id'], axis=1).columns)


Xtrain.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
Xtrain.head()


print("creating stats...")
labs_vitals_stats = Xtrain.groupby(['subject_id', 'hadm_id', 'icustay_id']).agg(['mean', 'std', 'count'])
labs_vitals_stats.columns = ['_'.join(col) for col in labs_vitals_stats.columns]

labs_vitals_stats = labs_vitals_stats.fillna(0)
labs_vitals_stats.head()


print("scaling")

scaler = StandardScaler()
X_train = scaler.fit_transform(labs_vitals_stats)

X_train = pd.DataFrame(X_train, columns=labs_vitals_stats.columns, index=labs_vitals_stats.index)
X_train.head()


number_of_clusters = clustering_helpers.elbow_method(X_train,10, 'stats')



print("clustering no time")
kmeans_labels, ward_labels, complete_labels, single_labels, labels_dict = run_algos(X_train, 2)
eval_metrics = pd.DataFrame(columns=["S_S", "PU"])

for keys, values in labels_dict.items():
    print("Clustering Algorithm: 16 windows", keys)
    s_s, pu = clustering_helpers.evaluation_metrics(X_train, values, ytrain.values, 'euclidean', rseed = RSEED)
    eval_metrics = eval_metrics.append({'S_S': s_s, 'PU': pu}, ignore_index=True)

    
    print("Silhouette Score: ", np.round(s_s, decimals=3) , "Purity: ", np.round(pu, decimals=3))
clustering_helpers.plot_clusters(X_train, kmeans_labels, ward_labels, complete_labels, single_labels,ytrain.values, 'no time')

eval_metrics.rename(index={0: 'Kmeans', 1:'Ward', 2:'Complete', 3: 'Single', 4: 'Spectral'}, inplace=True)
eval_metrics.to_csv("eval_metrics_no_time.csv")