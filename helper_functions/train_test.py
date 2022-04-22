import pandas as pd 
from helpers import *
from clustering_helpers import *

import warnings
warnings.filterwarnings("ignore")
RSEED = 2


# reading demographics and storing patients sets

demographics = pd.read_hdf('patient_clustering_EHR/data/data_icu.h5', key='demographics') 
#C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/patient_clustering_EHR/
labs_vitals = pd.read_hdf('C:/Users/Maria/Desktop/Work/Projects/MIMIC/Representation-Clustering/patient_clustering_EHR/labs_vitals.h5', key='preprocessed') 

labs = labs_vitals.reset_index()
#storing the true labels
true_labels = demographics[demographics.subject_id.isin(labs.subject_id)][['subject_id', 'hadm_id', 'icustay_id', 'mort_icu']]
del labs 


list_of_features = list(labs_vitals.drop(['hours_in', 'hours_in_'], axis=1).columns)

positive = (demographics.mort_icu.value_counts().values[0])
random_sample_positive = int(positive/5)
random_sample_positive_subj_id = demographics[demographics.mort_icu == 0].sample(n=random_sample_positive, random_state=RSEED).subject_id
negative_subj_id = demographics[demographics.mort_icu == 1].subject_id
labs_vitals.reset_index(inplace=True)
labs_vitals = labs_vitals[(labs_vitals.subject_id.isin(negative_subj_id.values)) | (labs_vitals.subject_id.isin(random_sample_positive_subj_id.values))]
labs_vitals.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
labs_vitals.head()
demographics.set_index("subject_id", inplace=True)


from sklearn.model_selection import train_test_split
demo_train, demo_test, y_train, y_test = train_test_split(demographics, demographics.mort_icu, test_size=0.9, random_state=42)
labs_vitals = labs_vitals.reset_index()
demo_train = demo_train.reset_index()
demo_test = demo_test.reset_index()

labs_train = labs_vitals[labs_vitals.subject_id.isin(demo_train.subject_id)]
y_train = y_train[y_train.index.isin(labs_train.subject_id)]
labs_test = labs_vitals[labs_vitals.subject_id.isin(demo_test.subject_id)]
y_test = y_test[y_test.index.isin(labs_test.subject_id)]


labs_train.to_hdf('patient_clustering_EHR/data/train_test_set.h5', key='Xtrain') 
y_train.to_hdf('patient_clustering_EHR/data/train_test_set.h5', key='ytrain') 
labs_test.to_hdf('patient_clustering_EHR/data/train_test_set.h5', key='Xtest') 
y_test.to_hdf('patient_clustering_EHR/data/train_test_set.h5', key='ytest') 
