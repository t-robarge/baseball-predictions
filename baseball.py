import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression, Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 0)  # Auto-adjust width to prevent line breaks
pd.set_option('display.expand_frame_repr', False)

# Load data
df = pd.read_csv('mlb_stats.csv')
df.set_index(df.columns[0], inplace=True)  # Assuming first column is team_id

# Extract y labels (end-of-season wins)
y_labels = df.pop('wins')

# Preserve mid-season wins before normalization
mid_season_wins = df['cur_wins'].copy()

# Normalize data
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

# Preserve team_id in train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled, y_labels, test_size=0.2, random_state=42
)

# Train model
lr = BayesianRidge()
lr.fit(X_train, y_train)
print(lr.coef_)
# Make predictions
y_pred = lr.predict(X_test)

# Convert results back to DataFrame
results_df = pd.DataFrame({
    'Actual Wins': y_test,
    'Predicted Wins': y_pred
}, index=X_test.index)

# Add mid-season wins to the results DataFrame
results_df['Mid-Season Wins'] = mid_season_wins.loc[X_test.index]

# Add difference column
results_df['RMSE'] = results_df['Actual Wins'] - results_df['Predicted Wins']

# Extract year from team_id (first 4 characters of the integer)
results_df['Year'] = results_df.index.astype(str).str[:4].astype(int)

# Sort by absolute difference
results_df = results_df.reindex(results_df['RMSE'].abs().sort_values(ascending=False).index)

# Print results
print(results_df)
print("RMSE:", mean_squared_error(y_test, y_pred)**(1/2))

# Calculate absolute difference and group by year
results_df['Absolute Difference'] = results_df['RMSE'].abs()
yearly_avg_abs_diff = results_df.groupby('Year')['Absolute Difference'].mean()
print("\nRMSE by Year:")
print(yearly_avg_abs_diff)