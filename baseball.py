import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load data
df = pd.read_csv('mlb_stats.csv')
df.set_index(df.columns[0], inplace=True)
#extract y labels
y_labels = df.pop('wins')
# normalize data
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

print(df_scaled.values.shape)
print(y_labels.shape)
X_train,X_test,y_train,y_test = train_test_split(df_scaled.values,y_labels,test_size=.2,random_state=42)
#lr = LinearRegression()
#lr = Ridge(alpha=1.0,random_state=42)
lr = BayesianRidge()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
side_by_side = np.column_stack((y_test,y_pred))
print(side_by_side)
print("Mean Squared Error:", mse)