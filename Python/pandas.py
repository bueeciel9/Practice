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