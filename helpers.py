import pandas as pd, numpy as np, datetime as dt




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




