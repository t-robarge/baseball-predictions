import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression
from sklearn.model_selection import test_train_split

df = pd.read_csv('mlb_stats.csv')

# Display the first few rows of the DataFrame to confirm it loaded correctly.
print(df.head())