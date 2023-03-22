# 1. Mount Drive

from google.colab import drive
drive.mount('/content/drive')

# 2. Load Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = '/gdrive/My Drive/data/TCS_영업소간통행시간_1시간_1개월_202001'
data = pd.read_csv(file, sep = ",", encoding = "euc-kr")

data.head()
data.tail()


# 3. Data Preprocessing

# 3-1. Data Cleaning

# 3-1-1. Check Null Values

data.isnull().sum()
data.columns
data_clean = data.dropna(axis = 0)
data_clean = data_clean[data_clean.통행시간>0] 


# select by dot operator
data.집계일자
# select by bracket operator
data["Unnamed: 6"]


# 4 Select Data
data_clean.head()
data_clean.columns

df_data = pd.DataFrame(data_clean, columns = ['집계일자', '집계시', '출발영업소코드'])
long_distance = df_data.통행시간 >700
df_data = df_data[df_data.통행시간 >700]

# 5. Insert Data
# The days are numbered from 0 to 6 where 0 is Monday and 6 is Sunday.
df_data['요일'] = pd.to_datetime(df_data['집계일자'], format = '%Y%m%d').dt.dayofweek





