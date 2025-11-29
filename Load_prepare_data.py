
import pandas as pd   # import pandas
import numpy as np
import seaborn as sb
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import join as pjoin

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset.csv')

# drop rows with missing values
df.isnull().sum()
df.dropna(inplace=True)
df = pd.DataFrame(df)

df.shape

df.head(10)
df.columns

df.describe().T


# Set the figure size
plt.figure(figsize=(35,35))
# Generate the heatmap
sb.heatmap(df.corr(),cmap='YlGnBu', annot=True)
#plt.savefig('Heatmap_FoodData.png')


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
df_scale = sc.fit_transform(df)

df_scale.shape