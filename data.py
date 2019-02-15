import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import util as u
import scipy.stats as stats
from sklearn.model_selection import train_test_split


# define functions
def fix_col_names(s):
    if type(s) == tuple:
        s = '_'.join(s)
    return s

def agg_mode(x):
    return stats.mode(x)[0][0]

def abs_max(x):
    return max(x, key=abs)

# import data
churn_df = pd.read_csv('Data/train_v2.csv')
trans_df = pd.read_csv('Data/transactions_v2.csv')
logs_df = pd.read_csv('Data/user_logs_v2.csv')
member_df = pd.read_csv('Data/members_v3.csv')

# create 'user_df': all info by user
user_df = pd.merge(churn_df, member_df, on='msno', how='left')

# Drop features that aren't useful
user_df.drop(columns=['gender', 'city', 'bd'], inplace=True)
user_df['registration_init_time'] = pd.to_datetime(user_df['registration_init_time'], format='%Y%m%d')
user_df['time_since_registration'] = round((dt.datetime(year=2017, month=4, day=30) - user_df['registration_init_time'])
                                 / dt.timedelta(days=30), 2)

# logs: summarize data
logs_df['num_songs'] = (logs_df['num_100'] + logs_df['num_25'] +
    logs_df['num_50'] + logs_df['num_75'] + logs_df['num_985'])
logs_df['full_pct'] = logs_df['num_100'] / logs_df['num_songs']
logs_df['short_pct'] = logs_df['num_25'] / logs_df['num_songs']
# Drop any entries whose total_seconds are greater than the # of seconds in a day
sec_in_day = 24*60*60
logs_df = logs_df[logs_df['total_secs'] < sec_in_day]
# Aggregate info in logs by user id
log_grp_df = logs_df.groupby('msno').aggregate({'num_25':'mean', 'num_50':'mean', 'num_75':'mean',
                                                'num_985':'mean', 'num_100':'mean',
                                                'num_unq':'mean', 'num_songs':['mean','sum'],
                                                'total_secs':['mean','sum','count'],
                                                'full_pct':'mean', 'short_pct':'mean'})
# fix multi-index column names
log_grp_df.columns = [fix_col_names(col) for col in log_grp_df.columns]
# merge with user_df
user_df = pd.merge(user_df, log_grp_df, how='left', on='msno')
user_df.rename(columns={'total_secs_count':'num_log_entries'}, inplace=True)


# TRANSACTIONs:

# dates
trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'], format='%Y%m%d')
trans_df['membership_expire_date'] = pd.to_datetime(trans_df['membership_expire_date'], format='%Y%m%d')
trans_df['transaction_month'] = pd.PeriodIndex(trans_df['transaction_date'], freq='M')
# convert payment_method_id to most common or not
trans_df['payment_method_most_common'] = trans_df['payment_method_id'].apply(lambda x: 1 if x == 41 else 0)
# calculate difference between plan price & actual amount paid
trans_df['plan_actual_diff'] = trans_df['plan_list_price'] - trans_df['actual_amount_paid']
# group first by user & transaction month
trans_grp_df = trans_df.groupby(['msno','transaction_month']).aggregate({'payment_method_most_common':agg_mode,
                                                               'payment_plan_days': agg_mode,
                                                                'plan_list_price': agg_mode,
                                                               'actual_amount_paid': agg_mode,
                                                               'is_auto_renew': 'max',
                                                               'is_cancel':'max',
                                                               'plan_actual_diff': [abs_max, agg_mode]})
# fix column names (resulting from multi-index)
trans_grp_df.columns = [fix_col_names(col) for col in trans_grp_df.columns]
trans_grp_df.reset_index(inplace=True)

# then group again by user
trans_grp2_df = trans_grp_df.groupby('msno').aggregate({'payment_method_most_common_agg_mode':agg_mode,
                                                'payment_plan_days_agg_mode': [agg_mode,'sum'],
                                                'plan_list_price_agg_mode': [agg_mode, 'sum'],
                                                'actual_amount_paid_agg_mode': [agg_mode, 'sum'],
                                                'is_auto_renew_max': [agg_mode, 'sum'],
                                                'is_cancel_max':[agg_mode, 'sum','count'],
                                                'plan_actual_diff_abs_max':abs_max,
                                                'plan_actual_diff_agg_mode':agg_mode})
# fix column names (resulting from multi-index)
trans_grp2_df.reset_index(inplace=True)
trans_grp2_df.columns = ['msno', 'payment_method_most_common_mode', 'payment_plan_days_mode',
                         'payment_plan_days_sum', 'plan_list_price_mode', 'plan_list_price_sum',
                         'actual_amount_paid_mode', 'actual_amount_paid_sum',
                         'is_auto_renew_mode', 'is_auto_renew_sum', 'is_cancel_mode','is_cancel_sum', 'trans_count',
                         'plan_actual_diff_abs_max', 'plan_actual_diff_mode']

# merge with user_df & save
user_df = pd.merge(user_df, trans_grp2_df, how='left', on='msno')
u.pkl_this('Data/user_df_from_script.pkl', user_df)
print('Done!')
# create train-test split
# target = 'is_churn'
# y = user_df[target]
# X = user_df.drop(columns=target)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
#
# u.pkl_this('Data/X_train.pkl', X_train)
# u.pkl_this('Data/X_test.pkl', X_test)
# u.pkl_this('Data/y_train.pkl', y_train)
# u.pkl_this('Data/y_test.pkl', y_test)
