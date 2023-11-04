import pandas as pd
import numpy as np
from tqdm import tqdm

# Load the dataset
file_path = 'path/Merged_df.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()

# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Check data types for each column
data_types = data.dtypes
missing_values, data_types

# Descriptive statistics for the dataset
descriptive_stats = data.describe()
descriptive_stats

from sklearn.preprocessing import StandardScaler
# Select only the numeric columns (excluding 'timestamp')
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Standardize the numeric columns
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_columns])

# Convert the scaled data back to a DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_columns)
# Display the first few rows of the scaled dataframe
data_scaled_df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# List of variables for univariate analysis
variables_to_plot = ['Price', 'Active Addresses', 'Tweets Volume', 'Tweets Sentiment']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=len(variables_to_plot), ncols=2, figsize=(15, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, var in enumerate(variables_to_plot):
    # Plot histogram
    sns.histplot(data_scaled_df[var], kde=True, ax=axes[i*2], color='skyblue')
    axes[i*2].set_title(f'Histogram of {var}')

    # Plot boxplot
    sns.boxplot(x=data_scaled_df[var], ax=axes[i*2 + 1], color='lightgreen')
    axes[i*2 + 1].set_title(f'Boxplot of {var}')

# Adjust the layout
plt.tight_layout()
plt.show()

# Calculate the Pearson correlation matrix
correlation_matrix = data_scaled_df[numeric_columns].corr()

# Plot a heatmap to visualize the correlation matrix
plt.figure(figsize=(16, 12))
heatmap = sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
plt.show()

# Extract the correlations with the Bitcoin price
btc_price_correlations = correlation_matrix['Price'].sort_values(ascending=False)
# Display the correlations with the Bitcoin price
btc_price_correlations

# Shift the 'Price' column up by one to predict the next day's price
data['Price_next_day'] = data['Price'].shift(-1)
# Drop the last row because it will have NaN as the 'Price_next_day'
data = data[:-1]

# Define features and target
X = data[["Price"]]  # Drop the current and next day price
y = data['Price_next_day']  # Predict the next day's price

# Split the data into training and testing sets without shuffling for a time series
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Display the size of each set to confirm our split
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize and train the Linear Regression Baseline model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lr_model.predict(X_test)

# Calculating error metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Print the performance metrics
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))  # Set the figure size as desired
plt.scatter(y_test, y_pred, alpha=0.5)  # Plot with a transparency of 0.5
plt.title('Actual vs Predicted Values')  # The title of the plot
plt.xlabel('Actual Values')  # The x-axis label
plt.ylabel('Predicted Values')  # The y-axis label

# Plotting a perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Show the plot
plt.show()

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

min_mae = 30000
for corr in tqdm([0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4]):
    # Prepare the features and target
    X = data_scaled_df.iloc[:, :-1][:-1]
    y = data['Price_next_day']

    to_drop = [column for column in correlation_matrix.columns if abs(correlation_matrix[column]["Price"]) < corr]
    X = X.drop(to_drop, axis=1)

    # Split the data into training and testing sets without shuffling for time series
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Display the size of each set to confirm our split
    (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = Ridge()
    parameters = {'alpha': np.logspace(-3, 3, 7)}

    grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    # Predicting the Test set results
    y_pred = best_model.predict(X_test)

    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    if mae < min_mae:
        min_mae = mae
        min_corr = corr

X = data_scaled_df.iloc[:, :-1][:-1]
y = data['Price_next_day']

to_drop = [column for column in correlation_matrix.columns if abs(correlation_matrix[column]["Price"]) < min_corr]
X = X.drop(to_drop, axis=1)

# Split the data into training and testing sets without shuffling for time series
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Let's display the size of each set to confirm our split
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Ridge()
parameters = {'alpha': np.logspace(-3, 3, 7)}

grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Predicting the Test set results
y_pred = best_model.predict(X_test)

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Print the performance metrics
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))  # Set the figure size as desired
plt.scatter(y_test, y_pred, alpha=0.5)  # Plot with a transparency of 0.5
plt.title('Actual vs Predicted Values')  # The title of the plot
plt.xlabel('Actual Values')  # The x-axis label
plt.ylabel('Predicted Values')  # The y-axis label

# Plotting a perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Show the plot
plt.show()