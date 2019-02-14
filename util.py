import pickle
import numpy as np
import pandas as pd

def pkl_this(filename, df):
    '''Saves df as filename. Must include .pkl extension'''
    with open(filename, 'wb') as picklefile:
        pickle.dump(df, picklefile)

def open_pkl(filename):
    '''Must include .pkl extension. Returns object that can be saved to a df.
    '''
    with open(filename,'rb') as picklefile:
        return pickle.load(picklefile)

def log_this(s):
    with open("log.txt", "a") as f:
        f.write("\n" + s + "\n")

def find_str_in_col(s, col, df):
    '''Finds string 's' in column 'col' of dataframe 'df'
    Input string in case format (lower, upper) that matches dataframe
    '''
    return df[df[col].str.contains(s)]
    # df[df['molecule_name'].str.contains('gilteritinib'.upper())]
    # return res


def percent_null(col, df):
    return len(df[df[col].isnull()]) / len(df) * 100

def to_num(col, df):
    df[col] = pd.to_numeric(df[col])
    return df
