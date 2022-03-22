import pandas as pd 

from helpers import *

from tslearn.metrics import cdist_dtw 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


RSEED = 2




def drop_recordings(df, threshold):
    """
    drop columns with recordings<threshold
    input -- 
        df: pandas df
        column_name: string, the name of multiindex column on level 1
        threshold
    output -- 
        df_red: pd.df, reduced df column 
    
    """
    #threshold = 0.99
    columns_to_drop = []
    for column_name in df.columns: 
        column = df[column_name]
        p = (column == 0).sum()/df.shape[0]
        if p > threshold:
            columns_to_drop.append(column_name)
    df_red = df.drop(columns=columns_to_drop)
    return df_red

def to_3d_numpy(df, columnA, columnW):
    x = df.reset_index()
    x.sort_values('subject_id', inplace=True)
    #x.drop(['hours_in', 'hours_in'], axis=1, inplace=True)
    count_id = x.subject_id.value_counts().sort_index().values
    mask = count_id[:,None] > np.arange(count_id.max())
    vals = x.loc[:, columnA:columnW].values
    out_shp = mask.shape + (vals.shape[1],)
    out = np.full(out_shp, np.nan)
    out[mask] = vals
    out.shape

    return out
    
def cut_time_series(df, time_window):
    print("cutting the series....")
    df_reset = df.reset_index()
    #del labs_vitals
    df_to_keep = []
    for i, icu, hadm in zip(set(df_reset.subject_id), set(df_reset.icustay_id), set(df_reset.hadm_id)):
        print(i)
        min_icu_patient_time = df_reset[df_reset.subject_id == i]['hours_in_'].min()
        #[labs_vitals_reset.subject_id == i[0]]
        max_icu_patient_time = df_reset[df_reset.subject_id == i]['hours_in_'].min() + time_window

        temp = df_reset[df_reset.subject_id == i]
        temp = temp[(temp['hours_in_'] >=min_icu_patient_time) & (temp['hours_in_'] <=max_icu_patient_time)]
            
        temp = temp.set_index('hours_in_').reindex(range(min_icu_patient_time, max_icu_patient_time)).reset_index()
        print(range(min_icu_patient_time, max_icu_patient_time))
            
        subject_id = [i] * len(set(temp.index))
        icustay_id = [icu] * len(set(temp.index))
        hadm_id = [hadm] * len(set(temp.index))

        hours_in = list(range(len(set(temp.index))))
        temp['hours_in_'] = hours_in
        temp['subject_id'] = subject_id
        temp['icustay_id'] = icustay_id
        temp['hadm_id'] = hadm_id

        df_to_keep.append(temp)

    final_df = pd.concat(df_to_keep)
    del df_to_keep

    final_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    final_df = final_df.sort_index(axis=0).sort_index(axis=1)

    return final_df

def purity(y_true, y_pred):
    
    """
    Input:
        y_true: numpy array, the true labels of your dataset 
        y_pred: numpy array, the labels predicted by the algorithm
    Output:
        purity: the resulting value for purity 

    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    # return purity
    return purity 



def pre_eucl(df):
    hours_in_ = df.hours_in_
    df = df.drop(['hours_in', 'hours_in_'], axis=1)

    out = to_3d_numpy(df, 'Alanine aminotransferase', 'pH')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(out.reshape(-1, out.shape[-1])).reshape(out.shape)

    m,n,r = X_train.shape
    out_arr = np.column_stack((np.repeat(np.arange(m),n),X_train.reshape(m*n,-1)))
    out_df = pd.DataFrame(out_arr, columns = df.columns.insert(0, 'drop-it'), index=df.index)
    out_df.drop('drop-it', axis=1, inplace=True)

    return out_df, X_train, hours_in_



def calc_euclideans(df, columnA, columnW):
    X = df.reset_index()
    X_ = X.loc[:, columnA:columnW]
    
    #'Alanine aminotransferase':'pH urine'
    df1  = (X.pivot(index='subject_id', columns='hours_in', values= X_.columns))

    cols_dict = {}
    for col in X.columns:
        if (col != 'hours_in') and (col != 'hours_in_') and (col != 'subject_id') and (col != 'hadm_id') and (col != 'icustay_id'):
            #arr1 = df1[col].to_numpy()
            cols_dict[col] =df1[col].to_numpy()
            #arr2 = df1['White blood cell count'].to_numpy()

    sum_ = np.empty_like(cols_dict['Albumin'])
    for keys, values in cols_dict.items():
        data = (values[None, :] - values[:, None])**2
        sum_ = sum_ + data 

    eucl = np.nanmean(np.sqrt((sum_)), axis=2)

    pairwise_euclidean = pd.DataFrame(eucl, index=df1.index, columns = df1.index)

    return pairwise_euclidean



def elbow_method(X, max_range_for_elbow, rseed = RSEED):
    """
    Input: 
        X: dataframe or numpy array, the dataset to be clustered, 
        max_range_for_elbow: int, the max number of clusters you want the elbow method to run. 
    """
    from yellowbrick.cluster import KElbowVisualizer

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,max_range_for_elbow+1))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    
    return visualizer.elbow_value_

def evaluation_metrics(X, labels_pred, labels_true,  metric, rseed = RSEED):
    """
    Input: 
        X: array-like, the values of the dataframe to be clustered
        labels_pred: numpy array, the labels predicted by the algorithm
        labels_true: numpy array, the ground truth/target/class
        metric: string, the metric to be used for silhouette score, example: 'euclidean'          

    Output:
        s_s: the value calculate by silhouette
        pu: the value calculated by the purity function
    
    """

    s_s = metrics.silhouette_score(X, labels_pred, metric=metric, random_state = rseed)
    pu = purity(labels_true, labels_pred)
    
    return s_s, pu


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

    spectral = SpectralClustering(n_clusters=number_of_clusters, affinity='precomputed')
    spectral_labels  = spectral.fit_predict(dtw_dist)
    labels_dict["spectral"] = spectral_labels

    return kmeans_labels, ward_labels, complete_labels, single_labels, spectral_labels, labels_dict

def plot_clusters(dtw_dist, kmeans_labels, ward_labels, complete_labels, single_labels, spectral_labels, true_labels):
    pca = PCA(n_components=2).fit(dtw_dist)
    pca_2d = pca.transform(dtw_dist)


    f, axes = plt.subplots(1, 6, sharey=True, figsize=(15, 6))


    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1], hue=true_labels, ax=axes[0])
    axes[0].set_title("Original" )

    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1],hue=kmeans_labels, ax=axes[1])
    axes[1].set_title("Kmeans")

    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1],hue=single_labels, ax=axes[2])
    axes[2].set_title("Agglomerative (Single)")

    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1],hue=ward_labels, ax=axes[3])
    axes[3].set_title("Agglomerative (Ward)")

    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1],hue=complete_labels, ax=axes[4])
    axes[4].set_title("Agglomerative (Complete)")

    sns.scatterplot(x=pca_2d[:, 0],y=pca_2d[:, 1],hue=spectral_labels, ax=axes[5])
    axes[5].set_title("Spectral")

    plt.tight_layout()
    plt.show()