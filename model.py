import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import re
import util as u


df_drop = u.open_pkl('Data/user_df_drop_na.pkl')

# train-test split
target = 'is_churn'
y = df_drop[target]
X = df_drop.drop(columns=target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

df_train = X_train.join(y_train)
df_train.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# FEATURE SELECTION & ENGINEERING

# PCA on correlated features:
pca_songs = PCA(n_components=1)
df_songs_pca = df_drop[['num_100_mean', 'num_unq_mean', 'num_songs_mean', 'num_songs_sum',
       'total_secs_mean', 'total_secs_sum']]
songs_pca = pca_songs.fit_transform(df_songs_pca)

pca_trans = PCA(n_components=1)
df_trans_pca = df_drop[['payment_plan_days_mode', 'payment_plan_days_sum',
       'plan_list_price_mode', 'plan_list_price_sum',
       'actual_amount_paid_mode', 'actual_amount_paid_sum']]
trans_pca = pca_songs.fit_transform(df_trans_pca)

# feature set 1
fs_cols_1 = ['registered_via',
             'short_pct_mean',
             'payment_method_most_common_mode',
             'plan_actual_diff_abs_max',
             'is_auto_renew_mode', 'is_cancel_mode',
             'trans_count',
             'time_since_registration']

df_feat1 = df_drop[fs_cols_1]
df_feat1 = df_feat1.join(pd.DataFrame(songs_pca, columns=['song_pca']))
df_feat1 = df_feat1.join(pd.DataFrame(trans_pca, columns=['transactions_pca']))


# PREPROCESSING PIPELINE:
# CatBoost only requires identification of Categorical features

# Categorical:
cat_cols = ['registered_via']
cat_cols_idx = [list(df_feat1.columns).index(x) for x in cat_cols]

# u.pkl_this('Data/X_train.pkl', X_train)
# u.pkl_this('Data/X_test.pkl', X_test)
# u.pkl_this('Data/y_train.pkl', y_train)
# u.pkl_this('Data/y_test.pkl', y_test)
