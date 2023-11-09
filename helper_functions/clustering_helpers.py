import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.manifold import TSNE


from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering, AgglomerativeClustering
from kneed import KneeLocator
from gower import gower_matrix
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.utils import resample
from scipy.optimize import linear_sum_assignment


def find_optimal_k_consensus(data, algorithm, max_k=10):
    # Perform the elbow method to find the optimal K for KMeans
    distortions = []
    for k in range(1, max_k + 1):
        if algorithm == 'KMeans':
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
        elif algorithm == 'KModes':
            kmodes = KModes(n_clusters=k, random_state=42)
            kmodes.fit(data)
            distortions.append(kmodes.cost_)

    # Find the "elbow" point where the distortion values start to level off
    kneedle = KneeLocator(range(1, max_k + 1), distortions, curve='convex', direction='decreasing')
    k_optimal = kneedle.elbow

    return k_optimal

def consensus(dataset_static, dataset_ts, algorithm_static, algorithm_ts, n_clusters_static, n_clusters_ts, num_iterations, optimize=True):


    dataset_static = dataset_static.astype(int)
    mixed_data = np.concatenate((dataset_static.values, dataset_ts), axis=1)
   


    num_samples = dataset_static.shape[0]
    # Perform Meta-CLustering Algorithm for Consensus (MCLA)
    consensus_matrix = np.zeros((num_samples, num_samples), dtype=int)
    # Automatically find the optimal K for KMeans using the elbow method
    if optimize == True:
        k_optimal_ts = find_optimal_k_consensus(dataset_ts, 'KMeans', max_k=10)
        k_optimal_static = find_optimal_k_consensus(dataset_static, 'KModes', max_k=10)
    else:
        k_optimal_ts = n_clusters_ts
        k_optimal_static = n_clusters_static


    print('optimal k ts: ', k_optimal_ts)
    print('optimal k static: ', k_optimal_ts)

    for _ in range(num_iterations):
        binary_clusterer = algorithm_static(n_clusters=k_optimal_static, random_state=np.random.randint(100))
        real_clusterer = algorithm_ts(n_clusters=k_optimal_ts, random_state=np.random.randint(100))

        binary_assignments = binary_clusterer.fit_predict(dataset_static)
        real_assignments = real_clusterer.fit_predict(dataset_ts)
        consensus_matrix += np.equal(binary_assignments[:, np.newaxis], real_assignments)


    # Normalize consensus matrix
    consensus_matrix = consensus_matrix/num_iterations


    optimal_num_clusters = find_optimal_k_consensus(consensus_matrix, 'KMeans', max_k=10)

    kmeans = KMeans(n_clusters=optimal_num_clusters)
    final_assignments = kmeans.fit_predict(consensus_matrix)


    return final_assignments

def compare_clustering_methods_baselines(dataset_static, dataset_ts, ts2, categorical_features_boolean_list,  dataset_name, max_clusters=10):
    """
    Compare different clustering methods: DBSCAN, hierarchical clustering with Gower distance, and K-Prototypes.

    Parameters:
        data (pd.DataFrame): The input data containing both binary and continuous features.
        categorical_features (list): List of indices (column numbers) of categorical features in the data.
        max_clusters (int): The maximum number of clusters to consider for K-Prototypes clustering.

    Returns:
        best_method (str): The name of the best clustering method ('DBSCAN', 'Hierarchical', or 'K-Prototypes').
        best_silhouette_score (float): The best silhouette score achieved among the methods.
        best_model: The best clustering model corresponding to the best silhouette score.
        results_df (pd.DataFrame): A DataFrame containing the clustering method names and their corresponding evaluation metrics.
    """
    
    # Preprocess the data to combine binary and continuous features into a single matrix
    # Combine binary and continuous features

    #time series should be false
    
    dataset_static = dataset_static.astype(int)
    columns = np.concatenate((dataset_static.columns, dataset_ts.columns), axis=None)
    # Combine binary and continuous features
    data = np.concatenate((dataset_static.values, ts2.values), axis=1)
    mixed_data = pd.DataFrame(data, columns = columns).values

    print(mixed_data)


    gower_distances_matrix = gower_matrix(mixed_data, cat_features=categorical_features_boolean_list)


    print('HDBSCAN')

    # Define the range of hyperparameters to search over
    param_grid = {
        'min_cluster_size': range(60, 90),
        'min_samples': range(2, 5)
    }

    hdbscan_best_score = -1.0
    best_hdbscan = None

    for min_cluster_size in param_grid['min_cluster_size']:
        for min_samples in param_grid['min_samples']:
            # Create an HDBSCAN instance with the current hyperparameters
            hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, metric='precomputed')
            #
            # Fit the model on the data
            hdbscan.fit(gower_distances_matrix.astype(np.float64))

            # Get the cluster labels
            hdbscan_labels = hdbscan.labels_
            
            # Check if more than one cluster is formed
            unique_clusters_hdbscan = np.unique(hdbscan_labels)
            if len(unique_clusters_hdbscan) > 1:

                # Calculate the silhouette score of the current model
                silhouette = silhouette_score(gower_distances_matrix, hdbscan_labels)

                # Count the number of outliers (-1 labels)
                num_outliers = np.sum(hdbscan_labels == -1)

                # Check if the current model has a higher silhouette score and fewer outliers than the best model so far

                if silhouette > hdbscan_best_score :
                    # and num_outliers < data.shape[0] * 0.1
                    #
                    #
                    hdbscan_best_score = silhouette
                    best_hdbscan = hdbscan

    # Get the best hyperparameters and the corresponding model
    best_params = {
        'min_cluster_size': best_hdbscan.min_cluster_size,
        'min_samples': best_hdbscan.min_samples
    }

    # Get the best cluster labels
    hdbscan_labels = best_hdbscan.labels_
    #Print the best hyperparameters and silhouette score
    print("Best Parameters:", best_params)
    print("Best Silhouette Score:", hdbscan_best_score)


    # Check if more than one cluster is formed
    unique_clusters_hdscan = np.unique(hdbscan_labels)

    if len(unique_clusters_hdscan) > 1:
        # Compute evaluation metrics
        silhouette_avg = silhouette_score(gower_distances_matrix, hdbscan_labels)
        calinski_harabasz_avg = calinski_harabasz_score(gower_distances_matrix, hdbscan_labels)
        davies_bouldin_avg = davies_bouldin_score(gower_distances_matrix, hdbscan_labels)

    print('K-PROTOTYPES')
    kprototypes_best_score = -1
    kprototypes_best_model = None

    index_of_categorical = pd.Series(categorical_features_boolean_list)

    ps_categorical = list(index_of_categorical[index_of_categorical==True].index)


    for n_clusters in range(2, max_clusters + 1):
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', verbose=0)
        clusters = kproto.fit_predict(mixed_data, categorical=ps_categorical)

        # Compute evaluation metrics
        silhouette_avg = silhouette_score(mixed_data, clusters)
        calinski_harabasz_avg = calinski_harabasz_score(mixed_data, clusters)
        davies_bouldin_avg = davies_bouldin_score(mixed_data, clusters)

        # Update the best model and score if necessary
        if silhouette_avg > kprototypes_best_score:
            kprototypes_best_score = silhouette_avg
            kprototypes_best_model = kproto


    print('HIERARCHICAL')
    # Perform hierarchical clustering with Gower distance
    hierarchical_best_score = -1
    hierarchical_best_model = None

    linkage_methods = ['single','complete', 'average', 'ward']
    # 

    for method in linkage_methods:
        linkage_matrix = linkage(gower_distances_matrix, method=method)
        for threshold in np.linspace(3, 10, num=10):
  
            clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

            # Check if more than one cluster is formed
            unique_clusters_hierarchical = np.unique(clusters)
            if len(unique_clusters_hierarchical) > 1:
                #print(len(unique_clusters_hierarchical))
                
                # Compute evaluation metrics
                silhouette_avg = silhouette_score(gower_distances_matrix, clusters)
                calinski_harabasz_avg = calinski_harabasz_score(gower_distances_matrix, clusters)
                davies_bouldin_avg = davies_bouldin_score(gower_distances_matrix, clusters)

                # Update the best model and score if necessary
                if silhouette_avg > hierarchical_best_score:
                    hierarchical_best_score = silhouette_avg
                    hierarchical_best_model = (linkage_matrix, threshold)

                    
    print('CONSENSUS')

    final_assignments_consensus = consensus(dataset_static, ts2, KModes, KMeans, None, None, 5, optimize=True)    

    silhouette_cc = silhouette_score(gower_distances_matrix, final_assignments_consensus)
    calinski_harabasz_cc = calinski_harabasz_score(gower_distances_matrix, final_assignments_consensus)
    davies_bouldin_cc = davies_bouldin_score(gower_distances_matrix, final_assignments_consensus)

    silhouette_cc_ts = silhouette_score(ts2, final_assignments_consensus)
    calinski_harabasz_cc_ts = calinski_harabasz_score(ts2, final_assignments_consensus)
    davies_bouldin_cc_ts = davies_bouldin_score(ts2, final_assignments_consensus)


    silhouette_cc_static = silhouette_score(dataset_static, final_assignments_consensus)
    calinski_harabasz_cc_static = calinski_harabasz_score(dataset_static, final_assignments_consensus)
    davies_bouldin_cc_static = davies_bouldin_score(dataset_static, final_assignments_consensus)
    # Perform K-Prototypes clustering
    

    # Compare the best scores of all methods
    best_silhouette_score = max(hdbscan_best_score, hierarchical_best_score, kprototypes_best_score, silhouette_cc)
    #hdbscan_best_score,
    # Determine the best method based on the best score
    if best_silhouette_score == hdbscan_best_score:
        best_method = 'HDBSCAN'
        best_model = best_hdbscan
        best_labels = hdbscan_labels

    elif best_silhouette_score == silhouette_cc:
        best_method = 'Consensus'
        best_model = 'Consensus'
        best_labels = final_assignments_consensus

    elif best_silhouette_score == hierarchical_best_score:
        best_method = 'Hierarchical'
        best_model =  'Hierarchical'
        best_labels = fcluster(hierarchical_best_model[0], t=hierarchical_best_model[1], criterion='distance')
    else:
        best_method = 'K-Prototypes'
        best_model = kprototypes_best_model
        best_labels = best_model.fit_predict(mixed_data, categorical=ps_categorical)
        

    # Create a DataFrame with the clustering method names and their evaluation metrics
    methods = ['HDBSCAN_gower', 'Hierarchical_gower', 'K-Prototypes', 'consensus', 'consensus_ts', 'consensus_static']
    silhouette_scores = [hdbscan_best_score, hierarchical_best_score, kprototypes_best_score, silhouette_cc, silhouette_cc_ts, silhouette_cc_static]
    #hdbscan_best_score
    calinski_harabasz_scores = [calinski_harabasz_avg, None, calinski_harabasz_avg, calinski_harabasz_cc, calinski_harabasz_cc_ts, calinski_harabasz_cc_static]
    davies_bouldin_scores = [davies_bouldin_avg, None, davies_bouldin_avg, davies_bouldin_cc, davies_bouldin_cc_ts, davies_bouldin_cc_static]
    results_df = pd.DataFrame({
        'Model': methods,
        'Silhouette Score': silhouette_scores,
        'Davies-Bouldin Score': davies_bouldin_scores,
        'Calinski-Harabasz Score': calinski_harabasz_scores,

    })


    num_clusters = len(np.unique(best_labels))
    print(num_clusters)
  

    best_reduced_data = TSNE(n_components=2, random_state=2, perplexity=50).fit_transform(gower_distances_matrix)

    
    plt.figure(figsize=(24, 18))
    for label in np.unique(best_labels):
        plt.scatter(best_reduced_data[best_labels == label, 0], best_reduced_data[best_labels == label, 1], label=f'Cluster {label}')
    plt.title(f'Best {best_model} Clustering - Dataset: {dataset_name}')
    plt.legend()


    return best_method, best_silhouette_score, best_model, results_df, best_labels


# Define the custom scoring function for silhouette score
def custom_silhouette_score(X, labels):
    try:
        return silhouette_score(X, labels)
    except ValueError:
        return -1.0  # Return a negative value in case of an error (e.g., when all samples are in one cluster)

def optimize_spectral_clustering(X, param_grid):
    best_silhouette_score = -1
    best_params = {}
    best_labels = None
    
    for n_clusters in param_grid['n_clusters']:
        for affinity in param_grid['affinity']:
            for n_neighbors in param_grid['n_neighbors']:
                spectral_clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    n_neighbors=n_neighbors
                )
                labels = spectral_clustering.fit_predict(X)
                silhouette_avg = silhouette_score(X, labels)
                
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_params = {
                        'n_clusters': n_clusters,
                        'affinity': affinity,
                        'n_neighbors': n_neighbors
                    }
                    best_labels = labels
                    
    return best_params, best_labels, best_silhouette_score



def find_optimal_k(data, max_k):
    # Perform the elbow method to find the optimal K for KMeans
    distortions = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    # Find the "elbow" point where the distortion values start to level off
    kneedle = KneeLocator(range(2, max_k + 1), distortions, curve='convex', direction='decreasing')
    k_optimal = kneedle.elbow

    return k_optimal


def optimize_agglomerative(data, max_k):
    from sklearn.cluster import AgglomerativeClustering

    m_clusters = np.arange(3, max_k + 1)
    best_silhouette_score = -1

    for cl in m_clusters:

        aggl = AgglomerativeClustering(n_clusters=cl, linkage='complete')
        aggl_labels = aggl.fit_predict(data)
        silhouette_agglomerative = silhouette_score(data, aggl_labels)

        if silhouette_agglomerative > best_silhouette_score:
            best_silhouette_score = silhouette_agglomerative
            best_params = {
                'n_clusters': cl,
            }
            best_labels = aggl_labels
    return best_labels



def compare_kmeans_hdbscan(data, dataset_name, max_k):
    # Automatically find the optimal K for KMeans using the elbow method
    k_optimal = find_optimal_k(data, max_k)

    # Perform KMeans and HDBSCAN clustering with the given parameters
    print('KMeans')
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)

    


    # optimize spectral 
    # Define parameter grid
    m_clusters = np.arange(5, max_k + 1)
    param_grid = {
        'n_clusters': m_clusters,
        'affinity': ['nearest_neighbors', 'rbf'],
        'n_neighbors': [5, 10, 15],
    }

    print('SPectral')

    #best_params_spectral, spectral_labels, best_silhouette_score = optimize_spectral_clustering(data, param_grid)
    #print(best_params_spectral)
    #spectral = SpectralClustering(n_clusters=best_params_spectral['n_clusters'], affinity=best_params_spectral['affinity'], n_neighbors=best_params_spectral['n_neighbors'] )
    #spectral_labels = spectral.fit_predict(data)
    spectral_labels = kmeans_labels

    # Define the range of hyperparameters to search over
    param_grid = {
        'min_cluster_size': range(40, 98),
        'min_samples': range(2, 5)
    }

    best_silhouette = -1.0
    best_hdbscan = None

    print('DBSCAN')
    for min_cluster_size in param_grid['min_cluster_size']:
        for min_samples in param_grid['min_samples']:
            # Create an HDBSCAN instance with the current hyperparameters
            hdbscan = HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples)
            #
            # Fit the model on the data
            hdbscan.fit(data)

            # Get the cluster labels
            hdbscan_labels = hdbscan.labels_

            # Calculate the silhouette score of the current model
            silhouette = silhouette_score(data, hdbscan_labels)
            #silhouette = davies_bouldin_score(data, hdbscan_labels)

            # Count the number of outliers (-1 labels)
            num_outliers = np.sum(hdbscan_labels == -1)

            # Check if the current model has a higher silhouette score and fewer outliers than the best model so far

            if silhouette > best_silhouette:
                #
                # and num_outliers < data.shape[0] * 0.1
                best_silhouette = silhouette
                best_hdbscan = hdbscan

    # Get the best hyperparameters and the corresponding model
    best_params = {
        'min_cluster_size': best_hdbscan.min_cluster_size,
        'min_samples': best_hdbscan.min_samples
    }

    print(best_params)

    # # Get the best cluster labels
    hdbscan_labels = best_hdbscan.labels_
    #hdbscan_labels = kmeans_labels
    print('Agglomerative')

    aggl_labels = optimize_agglomerative(data, max_k = max_k)
    # Calculate evaluation metrics
    silhouette_kmeans = silhouette_score(data, kmeans_labels)
    silhouette_hdbscan = silhouette_score(data, hdbscan_labels)
    silhouette_spectral = silhouette_score(data, spectral_labels)
    silhouette_aggl = silhouette_score(data, aggl_labels)
    
    db_score_kmeans = davies_bouldin_score(data, kmeans_labels)
    db_score_hdbscan = davies_bouldin_score(data, hdbscan_labels)
    db_score_spectral = davies_bouldin_score(data, spectral_labels)
    db_score_aggl = davies_bouldin_score(data, aggl_labels)
    
    ch_score_kmeans = calinski_harabasz_score(data, kmeans_labels)
    ch_score_hdbscan = calinski_harabasz_score(data, hdbscan_labels)
    ch_score_spectral = calinski_harabasz_score(data, spectral_labels)
    ch_score_aggl = calinski_harabasz_score(data, aggl_labels)

    # Create DataFrames for each set of labels
    kmeans_df = pd.DataFrame({f'{dataset_name}_KMeans': kmeans_labels})
    spectral_df = pd.DataFrame({f'{dataset_name}_Spectral': spectral_labels})
    hdbscan_df = pd.DataFrame({f'{dataset_name}_HDBSCAN': hdbscan_labels})
    aggl_df = pd.DataFrame({f'{dataset_name}_Agglomerative': aggl_labels})

    # Concatenate the DataFrames along columns
    labels_df = pd.concat([kmeans_df, spectral_df, hdbscan_df, aggl_df], axis=1)
    # Create a DataFrame to store the results
    metrics = {
        'Model': ['KMeans', 'HDBSCAN', 'Spectral', 'Agglomerative'],
        'Silhouette Score': [silhouette_kmeans, silhouette_hdbscan, silhouette_spectral, silhouette_aggl],
        'Davies-Bouldin Score': [db_score_kmeans, db_score_hdbscan, db_score_spectral, db_score_aggl],
        'Calinski-Harabasz Score': [ch_score_kmeans, ch_score_hdbscan, ch_score_spectral, ch_score_aggl]
    }
    df_results = pd.DataFrame(metrics)

    # Find the row with the maximum silhouette score
    max_silhouette_row = df_results[df_results['Silhouette Score'] == df_results['Silhouette Score'].max()]

    # Get the model name from the max_silhouette_row
    best_model = max_silhouette_row['Model'].values[0]
    best_labels = labels_df[f'{dataset_name}_{best_model}']

    print(f"The model with the maximum silhouette score is: {best_model}")




    if data.shape[1] > 2:
        best_reduced_data = TSNE(n_components=2, random_state=2, perplexity=50).fit_transform(data)   
    else:
        best_reduced_data = data


    # Plot both the best-performing KMeans and HDBSCAN
    plt.figure(figsize=(24, 18))
    # Plot best-performing KMeans
    plt.subplot(1, 4, 1)
    unique_labels_kmeans = np.unique(kmeans_labels)
    print(pd.Series(kmeans_labels).value_counts())
    num_clusters = len(unique_labels_kmeans)
    colors = plt.cm.get_cmap('tab20', num_clusters)
    for i, label in enumerate(unique_labels_kmeans):
        cluster_mask = (kmeans_labels == label)
        plt.scatter(best_reduced_data[cluster_mask, 0], best_reduced_data[cluster_mask, 1], color=colors(i), label=f'Cluster {label}', s=30)
    plt.title(f'Best KMEANS Clustering - Dataset: {dataset_name}')
    plt.legend()

    # Plot best-performing HDBSCAN
    plt.subplot(1, 4, 2)
    unique_labels_hdbscan = np.unique(hdbscan_labels)
    print(pd.Series(hdbscan_labels).value_counts())
    num_clusters = len(unique_labels_hdbscan)
    colors = plt.cm.get_cmap('tab20', num_clusters)
    for i, label in enumerate(unique_labels_hdbscan):
        cluster_mask = (hdbscan_labels == label)
        plt.scatter(best_reduced_data[cluster_mask, 0], best_reduced_data[cluster_mask, 1], color=colors(i), label=f'Cluster {label}', s=30)
    plt.title(f'Best HDBSCAN Clustering - Dataset: {dataset_name}')
    plt.legend()
    
    # Plot best-performing Spectral
    plt.subplot(1, 4, 3)
    unique_labels_spectral = np.unique(spectral_labels)
    print(pd.Series(spectral_labels).value_counts())
    num_clusters = len(unique_labels_spectral)
    colors = plt.cm.get_cmap('tab20', num_clusters)
    for i, label in enumerate(unique_labels_spectral):
        cluster_mask = (spectral_labels == label)
        plt.scatter(best_reduced_data[cluster_mask, 0], best_reduced_data[cluster_mask, 1], color=colors(i), label=f'Cluster {label}', s=30)
    plt.title(f'Best SPECTRAL Clustering - Dataset: {dataset_name}')
    plt.legend()

    # Plot both the best-performing KMeans and HDBSCAN
    plt.figure(figsize=(24, 18))
    # Plot best-performing KMeans
    plt.subplot(1, 4, 4)
    unique_labels_kmeans = np.unique(aggl_labels)
    print(pd.Series(aggl_labels).value_counts())
    num_clusters = len(unique_labels_kmeans)
    colors = plt.cm.get_cmap('tab20', num_clusters)
    for i, label in enumerate(unique_labels_kmeans):
        cluster_mask = (aggl_labels == label)
        plt.scatter(best_reduced_data[cluster_mask, 0], best_reduced_data[cluster_mask, 1], color=colors(i), label=f'Cluster {label}', s=30)
    plt.title(f'Best Aggloemrative Clustering - Dataset: {dataset_name}')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return df_results, best_labels, labels_df, best_model



def calculate_jaccard_index(labels1, labels2):
    """
    Calculate the Jaccard Index between two sets of cluster labels.

    Parameters:
        labels1 (numpy array): Cluster labels from the first run.
        labels2 (numpy array): Cluster labels from the second run.

    Returns:
        float: Jaccard Index value.
    """

    intersection = np.sum(labels1 == labels2)
    union = np.sum(np.logical_or(labels1 >= 0, labels2 >= 0))
    return intersection / union


def match_labels(labels1, labels2):
    """
    Match cluster labels between two runs using the Hungarian algorithm.

    Parameters:
        labels1 (numpy array): Cluster labels from the first run.
        labels2 (numpy array): Cluster labels from the second run.

    Returns:
        numpy array: Cluster labels from the second run, aligned with the first run.
    """

    num_labels1 = np.max(labels1) + 1
    num_labels2 = np.max(labels2) + 1

    label_overlap = np.zeros((num_labels1, num_labels2), dtype=int)

    for i in range(num_labels1):
        for j in range(num_labels2):
            label_overlap[i, j] = np.sum((labels1 == i) & (labels2 == j))

    row_ind, col_ind = linear_sum_assignment(-label_overlap)

    aligned_labels2 = np.zeros_like(labels2)
    for i, j in zip(row_ind, col_ind):
        aligned_labels2[labels2 == j] = i

    return aligned_labels2


def cluster_stability(data, original_cluster_labels, clusters, cl_algorithm = 'spectral', num_iterations=100):
    """
    Perform cluster stability analysis using HDBSCAN with bootstrap clustering and label matching.

    Parameters:
        data (numpy array): The input dataset.
        min_cluster_size (int): Minimum number of points required to form a cluster in HDBSCAN.
        num_iterations (int, optional): Number of bootstrap iterations. Default is 100.

    Returns:
        list of numpy arrays: Consistent cluster labels after matching.
        numpy array: Jaccard Index values across the bootstrap runs.
    """

    # Initialize an empty list to store bootstrap clusterings
    bootstrap_clusterings = []
    jaccard_indices = []
    resample_indices = []


    # Define the range of hyperparameters to search over
    param_grid = {
        'min_cluster_size': range(2, 40),
        'min_samples': range(2, 5)
    }

    # Perform bootstrap iterations
    for i in range(num_iterations):

        # Resample data with replacement to create a bootstrap sample

        bootstrap_indices = resample(np.arange(len(data)), n_samples = 800, random_state=i, replace=False) #bootstap
        #bootstrap_indices = shuffle(np.arange(len(data)), n_samples = 200, random_state=i) #pertubation
        #perturbed_data
        bootstrap_sample = data[bootstrap_indices]



        # best_silhouette = -1.0
        # best_hdbscan = None
        bootstrap_cluster_labels = None


        if cl_algorithm == 'hdbscan':
            # Define the range of hyperparameters to search over
            param_grid = {
                'min_cluster_size': range(50, 98),
                'min_samples': range(2, 5)
            }

            best_silhouette = -1.0
            best_hdbscan = None

            print('DBSCAN')
            for min_cluster_size in param_grid['min_cluster_size']:
                for min_samples in param_grid['min_samples']:
                    # Create an HDBSCAN instance with the current hyperparameters
                    hdbscan = HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples)
                    #
                    # Fit the model on the data
                    hdbscan.fit(bootstrap_sample)

                    # Get the cluster labels
                    hdbscan_labels = hdbscan.labels_

                    # Calculate the silhouette score of the current model
                    silhouette = silhouette_score(bootstrap_sample, hdbscan_labels)
                    #silhouette = davies_bouldin_score(data, hdbscan_labels)

                    # Count the number of outliers (-1 labels)
                    num_outliers = np.sum(hdbscan_labels == -1)

                    # Check if the current model has a higher silhouette score and fewer outliers than the best model so far

                    if silhouette > best_silhouette:
                        #
                        # and num_outliers < data.shape[0] * 0.1
                        best_silhouette = silhouette
                        best_hdbscan = hdbscan

            # Get the best hyperparameters and the corresponding model
            best_params = {
                'min_cluster_size': best_hdbscan.min_cluster_size,
                'min_samples': best_hdbscan.min_samples
            }

            #print(best_params)

            # # Get the best cluster labels
            hdbscan_labels = best_hdbscan.labels_

            

            # # Perform HDBSCAN clustering on the bootstrap sample
            model = HDBSCAN(min_cluster_size= best_params['min_cluster_size'], min_samples= best_params['min_samples'])
            labels = model.fit_predict(bootstrap_sample)

        elif cl_algorithm == 'spectral':
            model = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors', n_neighbors=15)
            labels = model.fit_predict(bootstrap_sample)
            
        elif cl_algorithm == 'kmeans':
            model = KMeans(n_clusters=clusters)
            labels = model.fit_predict(bootstrap_sample)
        elif cl_algorithm == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=clusters, linkage='ward')
            labels = model.fit_predict(bootstrap_sample)

        # Check if more than one cluster is formed
        unique_clusters = np.unique(labels)
        if len(unique_clusters) > 1:

            # Calculate the silhouette score of the current model
            silhouette = silhouette_score(bootstrap_sample, labels)


            # # Count the number of outliers (-1 labels)
            # num_outliers = np.sum(labels == -1)

            # Check if the current model has a higher silhouette score and fewer outliers than the best model so far

            #if silhouette > best_silhouette :
                #and num_outliers < bootstrap_sample.shape[0] * 0.1

            #best_silhouette = silhouette
            
        best_model = model
        bootstrap_cluster_labels = labels

        # Get the best cluster labels
        bootstrap_cluster_labels = best_model.labels_



        # Align the cluster labels between runs using the Hungarian algorithm
        aligned_labels = match_labels(original_cluster_labels[bootstrap_indices], bootstrap_cluster_labels)


        # Calculate Jaccard Index for each run compared to the first run
        jaccard_index = calculate_jaccard_index(original_cluster_labels[bootstrap_indices], aligned_labels)
        jaccard_indices.append(jaccard_index)


    return np.array(jaccard_indices)