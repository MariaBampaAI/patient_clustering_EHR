import pandas as pd 
import numpy as np
import helper_functions.helpers as helpers
import warnings
warnings.filterwarnings("ignore")


# reading demographics and storing patients sets

demographics = pd.read_hdf('D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\data_latest\\demographics.h5', key='demographics') 
subject_id_set = set(demographics['subject_id'])
hadm_id_set = set(demographics['hadm_id'])
icustay_id_set = set(demographics['icustay_id'])

print("Variable mapping")
# get the variable mapping
var_map = helpers.get_variable_mapping("patient_clustering_EHR\\resources\\itemid_to_variable_map.csv")
chartitems_to_keep = set(var_map.loc[var_map['LINKSTO'] == 'chartevents'].index)
labitems_to_keep = set(var_map.loc[var_map['LINKSTO'] == 'labevents'].index)
#patient_clustering_EHR\\
# read labs 
print("Reading labs...")
labs_vitals = pd.read_hdf('D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\data_latest\\data_icu.h5', key='labs_vitals')
labs_vitals.to_hdf('D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\data_latest\\labs_vitals.h5', key='labs_vitals') 

labs_vitals.reset_index(inplace=True)
#storing the true labels
true_labels = demographics[demographics.subject_id.isin(labs_vitals.subject_id)][['subject_id', 'hadm_id', 'icustay_id', 'mort_icu']]

print("Hourly buckets")
#### hourly buckets ####################################
to_hours = lambda x: max(0, x.days*24 + x.seconds // 3600)
#  the value is str, convert to numeric
labs_vitals['value'] = pd.to_numeric(labs_vitals['value'], 'coerce')
# join and add in labs_vital the icu intime and outime, to separate in hourly buckets
demographics = demographics.set_index("icustay_id")
labs_vitals = labs_vitals.set_index('icustay_id').join(demographics[['intime', 'outtime']])
# to hourly buckets
labs_vitals['hours_in'] = (labs_vitals['charttime'] - labs_vitals['intime']).apply(to_hours)


# to filter the itemids table
itemids = set(labs_vitals.itemid.astype(str))

"""

connection = psycopg2.connect(
    user = 'postgres',
    database="mimic",
    #password=getpass.getpass("Enter postgres password"), 
    password="EMm=N*]9}4yE",
    host="127.0.0.1", 
    port = "5433",
    options=f'-c search_path=mimiciii')



query_d_items = \

SELECT itemid, label, dbsource, linksto, category, unitname
FROM d_items
WHERE itemid in ({itemids})
;
.format(itemids=','.join(itemids))
items_ids = pd.read_sql_query(query_d_items, connection).set_index('itemid')
"""
items_ids = pd.read_csv("D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\resources\\items_ids.csv")
items_ids = items_ids.set_index('itemid')


print(items_ids)

items_ids.to_csv("patient_clustering_EHR\\resources\\items_ids.csv")




labs_vitals.drop(columns=['charttime', 'intime', 'outtime'], inplace=True)
labs_vitals.set_index('itemid', append=True, inplace=True)
labs_vitals =  labs_vitals.join(var_map).join(items_ids)

print("Feature Standardization")
## standardize features like temperature.

labs_vitals = helpers.standardize_units(labs_vitals)

print("Variable ranges")
#variable ranges
var_ranges = helpers.get_variable_ranges("D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\resources\\variable_ranges.csv")
labs_vitals.set_index(['label', 'LEVEL2'], append=True,inplace=True)
labs_vitals = helpers.apply_variable_limits(labs_vitals, var_ranges, 'LEVEL2')
#labs_vitals_c = labs_vitals.copy()

print("Aggregate and get stats")
#labs_vitals_stats = labs_vitals.groupby(['subject_id', 'hadm_id', 'icustay_id'] + ['LEVEL2'] + ['hours_in']).agg(['mean', 'std', 'count'])
labs_vitals = labs_vitals.groupby(['subject_id', 'hadm_id', 'icustay_id'] + ['LEVEL2'] + ['hours_in']).agg(['mean'])
labs_vitals.drop('index', axis=1, level=0, inplace=True)
labs_vitals.columns = labs_vitals.columns.droplevel(0)
labs_vitals.columns.names = ['Aggregated']


demographics['max_hours'] = (demographics['outtime'] - demographics['intime']).apply(to_hours)
# Pivot table droups NaN columns so you lose any uniformly NaN.
labs_vitals = labs_vitals.unstack(level = ['LEVEL2'])
labs_vitals.columns = labs_vitals.columns.reorder_levels(order=['LEVEL2'] + ['Aggregated'])
labs_vitals = labs_vitals.sort_index(axis=0).sort_index(axis=1)

labs_vitals = labs_vitals.T.groupby(level=0).first().T

labs_vitals.to_hdf('D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\data_latest\\labs_vitals.h5', key='preprocessed_all_ts') 
#keep 48h
print("keeping 48h....")
labs_vitals_reset = labs_vitals.reset_index()
del labs_vitals
df_to_keep = []
patients_less_48 = 0
patients_more_48 = 0
for i, icu, hadm in zip(set(labs_vitals_reset.subject_id), set(labs_vitals_reset.icustay_id), set(labs_vitals_reset.hadm_id)):
    print(i)
    max_icu_patient_time = labs_vitals_reset[labs_vitals_reset.subject_id == i]['hours_in'].max()
    #[labs_vitals_reset.subject_id == i[0]]
    min_icu_patient_time = labs_vitals_reset[labs_vitals_reset.subject_id == i]['hours_in'].max() - 48
    #print(max_icu_patient_time, min_icu_patient_time)
    if max_icu_patient_time >= 48:

        patients_more_48 = patients_more_48 + 1
        print(">48h", i)
        temp = labs_vitals_reset[labs_vitals_reset.subject_id == i]
        temp = temp[(temp['hours_in'] >=min_icu_patient_time) & (temp['hours_in'] <=max_icu_patient_time)]
        
        #temp = temp.set_index('hours_in').reindex(range(temp['hours_in'].min(), temp['hours_in'].max())).reset_index()
        temp = temp.set_index('hours_in').reindex(range(min_icu_patient_time, max_icu_patient_time)).reset_index()
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
    else:
        patients_less_48 = patients_less_48 + 1
        print("<48h", i)

print("len of patient with more than 48", patients_more_48 )
print("len of patient with less than 48", patients_less_48 )


vital_labs_48 = pd.concat(df_to_keep)
del df_to_keep

# Do linear interpolation where missingness is low (kNN imputation doesnt work if all rows have missing values)
print('Full ICU -- Doing linear interpolation where missingness is low (kNN imputation doesnt work if all rows have missing values)')
miss = np.sum(np.isnan(vital_labs_48), axis=0)/vital_labs_48.shape[0]
#ii = (miss>0) & (miss<0.05)  #less than 5% missingness
#mechventcol = reformat3t.columns.tolist().index('mechvent')
print(miss)


vital_labs_48.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
vital_labs_48 = vital_labs_48.sort_index(axis=0).sort_index(axis=1)


########################################################################
#             HANDLING OF MISSING VALUES  &  CREATE REFORMAT4T
########################################################################
#df.interpolate(limit=1, limit_direction="forward");

#imputer = KNNImputer(n_neighbors=1, weights='uniform', metric='nan_euclidean')

imputed_to_keep = []
for i in range(0,  vital_labs_48.shape[0], 48):
    temp = vital_labs_48.loc[[vital_labs_48.index[i][0]]]
    for col, val in temp.iteritems():
        #print(col)
        if (col != 'hours_in') and (col != 'hours_in_') and (~np.all(temp[col].isna())):
            #(str(col[1]) == 'mean') and 
            print('patient', vital_labs_48.index[i][0], 'feature', col)
            t = np.array(temp[col]).reshape(-1, 1)

            #imputed_col = imputer.fit_transform(t)
            imputed_col = temp[col].interpolate()
            temp.drop(col, axis=1, inplace=True)
            temp[col] = imputed_col
    imputed_to_keep.append(temp)

vital_labs_48_imp = pd.concat(imputed_to_keep)

del imputed_to_keep
#vital_labs_48_imp = vital_labs_48_imp.fillna(0)

vital_labs_48_imp.to_hdf('D:\\Projects\\MIMIC\\Representation-Clustering\\patient_clustering_EHR\\data_latest\\labs_vitals.h5', key='preprocessed_48h') 

