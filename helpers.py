import pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt 




def str2time(val):
    """
    Convert str to pandas datetime

    Input: val, pandas column
    Output: pd.Series converted to datetime or Nat 
    
    """
    try:
        return dt.datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
    except:
        return pd.NaT




def drop_recordings(df, column_name_, threshold):
    """
    drop columns with recordings<threshold
    input -- 
        df: pandas df
        column_name: string, the name of multiindex column on level 1
        threshold
    output -- 
        df_red: pd.df, reduced df column 
    
    """
    threshold = 0.9
    columns_to_drop = []
    for column_name in df.columns:
        #print(column_name)
        if column_name[1] == column_name_:
            #get the only the means of the vital_labs
            if df[column_name].isnull().mean() > threshold:
                columns_to_drop.append(column_name[0])
    df_red = df.drop(columns=columns_to_drop)
    return df_red



# From MIMIC extract paper
def get_variable_mapping(mimic_mapping_filename):
    # Read in the second level mapping of the itemids
    var_map = pd.read_csv(mimic_mapping_filename, index_col=None)
    var_map = var_map.loc[(var_map['LEVEL2'] != '') & (var_map['COUNT']>0) & (var_map['STATUS'] == 'ready')]
    var_map['ITEMID'] = var_map['ITEMID'].astype(int)
    # renaming to match the mimic tables
    var_map.rename(columns={'ITEMID': 'itemid'}, inplace=True)
    var_map = var_map[['LEVEL2', 'itemid', 'LEVEL1', 'LINKSTO']].set_index('itemid')
    

    return var_map





def get_variable_ranges(range_filename):
    # Read in the second level mapping of the itemid, and take those values out
    columns = [ 'LEVEL2', 'OUTLIER LOW', 'VALID LOW', 'IMPUTE', 'VALID HIGH', 'OUTLIER HIGH' ]
    to_rename = dict(zip(columns, [ c.replace(' ', '_') for c in columns ]))
    to_rename['LEVEL2'] = 'VARIABLE'
    var_ranges = pd.read_csv(range_filename, index_col=None)
    var_ranges = var_ranges[columns]
    var_ranges.rename(columns=to_rename, inplace=True)
    var_ranges = var_ranges.drop_duplicates(subset='VARIABLE', keep='first')
    var_ranges['VARIABLE'] = var_ranges['VARIABLE'].str.lower()
    var_ranges.set_index('VARIABLE', inplace=True)
    var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]

    return var_ranges

def apply_variable_limits(df, var_ranges, var_names_index_col='LEVEL2'):
    idx_vals        = df.index.get_level_values(var_names_index_col)
    non_null_idx    = ~df.value.isnull()
    var_names       = set(idx_vals)
    var_range_names = set(var_ranges.index.values)

    for var_name in var_names:
        var_name_lower = var_name.lower()
        if var_name_lower not in var_range_names:
            print("No known ranges for %s" % var_name)
            continue

        outlier_low_val, outlier_high_val, valid_low_val, valid_high_val = [
            var_ranges.loc[var_name_lower, x] for x in ('OUTLIER_LOW','OUTLIER_HIGH','VALID_LOW','VALID_HIGH')
        ]

        running_idx = non_null_idx & (idx_vals == var_name)

        outlier_low_idx  = (df.value < outlier_low_val)
        outlier_high_idx = (df.value > outlier_high_val)
        valid_low_idx    = ~outlier_low_idx & (df.value < valid_low_val)
        valid_high_idx   = ~outlier_high_idx & (df.value > valid_high_val)

        var_outlier_idx   = running_idx & (outlier_low_idx | outlier_high_idx)
        var_valid_low_idx = running_idx & valid_low_idx
        var_valid_high_idx = running_idx & valid_high_idx

        df.loc[var_outlier_idx, 'value'] = np.nan
        df.loc[var_valid_low_idx, 'value'] = valid_low_val
        df.loc[var_valid_high_idx, 'value'] = valid_high_val

        n_outlier = sum(var_outlier_idx)
        n_valid_low = sum(var_valid_low_idx)
        n_valid_high = sum(var_valid_high_idx)
        if n_outlier + n_valid_low + n_valid_high > 0:
            print(
                "%s had %d / %d rows cleaned:\n"
                "  %d rows were strict outliers, set to np.nan\n"
                "  %d rows were low valid outliers, set to %.2f\n"
                "  %d rows were high valid outliers, set to %.2f\n"
                "" % (
                    var_name,
                    n_outlier + n_valid_low + n_valid_high, sum(running_idx),
                    n_outlier, n_valid_low, valid_low_val, n_valid_high, valid_high_val
                )
            )

    return df


def get_values_by_name_from_df_column_or_index(data_df, colname):
    """ Easily get values for named field, whether a column or an index
    Returns
    -------
    values : 1D array
    """
    try:
        values = data_df[colname]
    except KeyError as e:
        if colname in data_df.index.names:
            values = data_df.index.get_level_values(colname)
        else:
            raise e
    return values


    
UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('Fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('Oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('Temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]
def standardize_units(X, name_col='LEVEL2', unit_col='valueuom', value_col='value', inplace=True):
    if not inplace: X = X.copy()
    name_col_vals = get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.str
        unit_col_vals = unit_col_vals.str
    except:
        print("Can't call *.str")
        print(name_col_vals)
        print(unit_col_vals)
        raise

    #name_filter, unit_filter = [
    #    (lambda n: col.contains(n, case=False, na=False)) for col in (name_col_vals, unit_col_vals)
    #]
    # TODO(mmd): Why does the above not work, but the below does?
    name_filter = lambda n: name_col_vals.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = name_filter_idx & False

        if unit is not None: needs_conversion_filter_idx |= name_filter(unit) | unit_filter(unit)
        if rng_check_fn is not None: needs_conversion_filter_idx |= rng_check_fn(X[value_col])

        idx = name_filter_idx & needs_conversion_filter_idx

        X.loc[idx, value_col] = convert_fn(X[value_col][idx])

    return X

def range_unnest(df, col, out_col_name=None, reset_index=False):
    assert len(df.index.names) == 1, "Does not support multi-index."
    if out_col_name is None: out_col_name = col

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].iteritems() for x in range(y+1)],
        columns=[df.index.names[0], out_col_name]
    )

    if not reset_index: col_flat = col_flat.set_index(df.index.names[0])
    return col_flat


def print_proportions(df):
    """
    # Print the total proportions!

    input: pandas df
    
    """


    rows, vars = df.shape
    print('')
    for l, vals in df.iteritems():
        if str(l[1]) == 'mean':
            ratio = 1.0 * vals.dropna().count() / rows
            print(str(l) + ': ' + str(round(ratio, 3)*100) + '% present')

    return None 




def plot_variable_histograms(col_names, df):
    # Plot some of the data, just to make sure it looks ok
    for c, vals in df.iteritems():
        n = vals.dropna().count()
        if n < 2: continue

        # get median, variance, skewness
        med = vals.dropna().median()
        var = vals.dropna().var()
        skew = vals.dropna().skew()

        # plot
        fig = plt.figure(figsize=(13, 6))
        plt.subplots(figsize=(13,6))
        vals.dropna().plot.hist(bins=100, label='HIST (n={})'.format(n))

        # fake plots for KS test, median, etc
        plt.plot([], label=' ',color='lightgray')
        plt.plot([], label='Median: {}'.format(format(med,'.2f')),
                 color='lightgray')
        plt.plot([], label='Variance: {}'.format(format(var,'.2f')),
                 color='lightgray')
        plt.plot([], label='Skew: {}'.format(format(skew,'.2f')),
                 color='lightgray')

        # add title, labels etc.
        plt.title('{} measurements in ICU '.format(str(c)))
        plt.xlabel(str(c))
        plt.legend(loc="upper left", bbox_to_anchor=(1,1),fontsize=12)
        plt.xlim(0, vals.quantile(0.99))
        #fig.savefig(os.path.join(outPath, (str(c) + '_HIST_.png')), bbox_inches='tight')



"""
itemid_to_variable_map = pd.read_csv("itemid_to_variable_map.csv")
item_ids_mapping = itemid_to_variable_map.groupby("LEVEL2")['ITEMID'].apply(list).to_dict()
print(len(item_ids_mapping))
remove_list = ['Blood culture',  'Cardiac Index', 'Cardiac Murmur', 'Cholesterol Pleural', 'Code Status','Consciousness Level',
'Ectopy Frequency','Ectopy Type','Fall Risk','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale verbal response','Glucose urine','Heart Rhythm','Lung Sounds',
'Orientation','Pacemaker','Pupillary response left','Pupillary response right','Pupillary size left','Pupillary size right','Riker-SAS Scale','Service Type','Skin Color','Skin Integrity',
'Total Protein Body Fluid','Total Protein Joint Fluid','Trach Size','Urine Appearance','Urine Color','Urine output','Ventilator Mode','Ventilator Type', "Calcium ionized"]

item_ids_mapping_red = dict([(key, val) for key, val in item_ids_mapping.items() if key not in remove_list])

print(len(item_ids_mapping_red))
item_ids_mapping_red["Calcium Ionized"] = [3766, 50808, 816, 225667]

import itertools
vitals_labs_to_keep_list = list(itertools.chain(*item_ids_mapping_red.values()))


"""