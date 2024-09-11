import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the data
file_red = r'C:\Users\TAN\Desktop\Level2_AI\machine_learning\CIS_grkp39\winequality-red.csv'
file_white = r'C:\Users\TAN\Desktop\Level2_AI\machine_learning\CIS_grkp39\winequality-white.csv' 

data_red = pd.read_csv(file_red, sep=';')
data_white = pd.read_csv(file_white, sep=';')

data_red_cleaned = data_red
data_white_cleaned = data_white

# Function to detect and print outliers
def detect_and_print_outliers(df):
    outlier_indices = []

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_list_col = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = list(set(outlier_indices))
    outliers = df.iloc[outlier_indices]
    print(f"Number of outliers: {len(outliers)}")
    print("Outliers:\n", outliers)

# Detect and print outliers for red wine
print("Red Wine Outliers:")
detect_and_print_outliers(data_red_cleaned)

# Detect and print outliers for white wine
print("\nWhite Wine Outliers:")
detect_and_print_outliers(data_white_cleaned)

# Add a new column for each dataset to differentiate between red and white wine
data_red['wine_type'] = 'Red'
data_white['wine_type'] = 'White'

# Merge two datasets
combined_data = pd.concat([data_red, data_white])

# Set the size of the drawing
plt.figure(figsize=(20, 10))

# Create violin maps for each feature
for i, col in enumerate(data_red.columns[:-1]):
    plt.subplot(3, 4, i + 1) 
    sns.violinplot(x='wine_type', y=col, data=combined_data)
    plt.title(col)

plt.tight_layout()
plt.show()

# Assuming 'wine_type' column is present or any non-numeric columns, drop them
data_red_numeric = data_red.select_dtypes(include=[float, int])
data_white_numeric = data_white.select_dtypes(include=[float, int])

# Calculate the correlation matrices
corr_red = data_red_numeric.corr()
corr_white = data_white_numeric.corr()

# Plotting the heatmap for Red Wine
plt.figure(figsize=(12, 10))
plt.title('Correlation Matrix of Red Wine Attributes')
sns.heatmap(corr_red, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Plotting the heatmap for White Wine
plt.figure(figsize=(12, 10))
plt.title('Correlation Matrix of White Wine Attributes')
sns.heatmap(corr_white, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# One-hot encode categorical variables (if any)
X_red = pd.get_dummies(data_red.drop('quality', axis=1))
y_red = data_red['quality']
X_white = pd.get_dummies(data_white.drop('quality', axis=1))
y_white = data_white['quality']

# It's also a good practice to scale features before fitting regression models
scaler = StandardScaler()
X_red_scaled = scaler.fit_transform(X_red)
X_white_scaled = scaler.fit_transform(X_white)

# Split the datasets into training and testing sets
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red_scaled, y_red, test_size=0.3, random_state=42)
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white_scaled, y_white, test_size=0.3, random_state=42)

# Fit the linear regression models
lr_red = LinearRegression()
lr_red.fit(X_train_red, y_train_red)
lr_white = LinearRegression()
lr_white.fit(X_train_white, y_train_white)

# Define the parameter grid for the decision tree model
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Fit the decision tree regressor models
dt_red = DecisionTreeRegressor(random_state=42)
dt_red.fit(X_train_red, y_train_red)
dt_white = DecisionTreeRegressor(random_state=42)
dt_white.fit(X_train_white, y_train_white)

# Create GridSearchCV objects, using 5-fold cross validation
grid_search_red = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_white = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')

# Grid search on red wine data
grid_search_red.fit(X_train_red, y_train_red)

# Grid searches on white wine data
grid_search_white.fit(X_train_white, y_train_white)

# Get the best parameters and performance on the test set
best_params_red = grid_search_red.best_params_
best_estimator_red = grid_search_red.best_estimator_
best_params_white = grid_search_white.best_params_
best_estimator_white = grid_search_white.best_estimator_

# Predictions using optimal parameters for the test set
y_pred_best_red = best_estimator_red.predict(X_test_red)
y_pred_best_white = best_estimator_white.predict(X_test_white)

# Make predictions
y_pred_lr_red = lr_red.predict(X_test_red)
y_pred_dt_red = dt_red.predict(X_test_red)
y_pred_lr_white = lr_white.predict(X_test_white)
y_pred_dt_white = dt_white.predict(X_test_white)

# Calculate mean squared error
mse_lr_red = mean_squared_error(y_test_red, y_pred_lr_red)
mse_dt_red = mean_squared_error(y_test_red, y_pred_dt_red)
mse_lr_white = mean_squared_error(y_test_white, y_pred_lr_white)
mse_dt_white = mean_squared_error(y_test_white, y_pred_dt_white)
rmse_best_red = np.sqrt(mean_squared_error(y_test_red, y_pred_best_red))
rmse_best_white = np.sqrt(mean_squared_error(y_test_white, y_pred_best_white))
rmse_lr_red = np.sqrt(mse_lr_red)
rmse_dt_red = np.sqrt(mse_dt_red)
rmse_lr_white = np.sqrt(mse_lr_white)
rmse_dt_white = np.sqrt(mse_dt_white)

# Create a DataFrame to store RMSE values
rmse_values = {
    'Model Type': ['Linear Regression', 'Decision Tree', 'Linear Regression', 'Decision Tree'],
    'RMSE': [rmse_lr_red, rmse_dt_red, rmse_lr_white, rmse_dt_white],
    'Wine Type': ['Red', 'Red', 'White', 'White']
}
rmse_df = pd.DataFrame(rmse_values)

# Plotting histograms of RMSE for comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model Type', y='RMSE', hue='Wine Type', data=rmse_df)
plt.title('RMSE of Linear Regression and Decision Tree Models for Different Wine Types')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xlabel('Model Type')
plt.legend(title='Wine Type')
plt.tight_layout()
plt.show()

# Predictions on test sets
y_pred_lr_red = lr_red.predict(X_test_red)
y_pred_dt_red = dt_red.predict(X_test_red)
y_pred_lr_white = lr_white.predict(X_test_white)
y_pred_dt_white = dt_white.predict(X_test_white)

# Convert regression output to categorical output, assuming a threshold of 6, i.e., quality scores greater than or equal to 7 are considered high quality
y_pred_lr_red_class = (y_pred_lr_red >= 6).astype(int)
y_pred_dt_red_class = (y_pred_dt_red >= 6).astype(int)
y_pred_lr_white_class = (y_pred_lr_white >= 6).astype(int)
y_pred_dt_white_class = (y_pred_dt_white >= 6).astype(int)

# Calculation accuracy
accuracy_lr_red = accuracy_score(y_test_red >= 6, y_pred_lr_red_class)
accuracy_dt_red = accuracy_score(y_test_red >= 6, y_pred_dt_red_class)
accuracy_lr_white = accuracy_score(y_test_white >= 6, y_pred_lr_white_class)
accuracy_dt_white = accuracy_score(y_test_white >= 6, y_pred_dt_white_class)

# Create a DataFrame to store the accuracy of each model and wine type combination
accuracy_data = pd.DataFrame({
    'Model Type': ['Linear Regression', 'Decision Tree', 'Linear Regression', 'Decision Tree'],
    'Accuracy': [accuracy_lr_red, accuracy_dt_red, accuracy_lr_white, accuracy_dt_white],
    'Wine Type': ['Red', 'Red', 'White', 'White']
})

# Histograms of accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x='Model Type', y='Accuracy', hue='Wine Type', data=accuracy_data)
plt.title('Accuracy of Models for Red and White Wines')
plt.ylabel('Accuracy (%)')
plt.xlabel('Model Type')
plt.legend(title='Wine Type')
plt.tight_layout()
plt.show()

# Print results
print("Red Wine:")
print("Linear Regression MSE:", mse_lr_red)
print("Decision Tree Regression MSE:", mse_dt_red)
print("Linear Regression RMSE:", rmse_lr_red)
print("Decision Tree Regression RMSE:", rmse_dt_red)
print(f'Linear Regression Accuracy for Red Wine: {accuracy_lr_red:.2%}')
print(f'Decision Tree Accuracy for Red Wine: {accuracy_dt_red:.2%}')
print("\nBest Decision Tree Model for Red Wine:")
print("Best Parameters:", best_params_red)
print("Test RMSE:", rmse_best_red)

print("\nWhite Wine:")
print("Linear Regression MSE:", mse_lr_white)
print("Decision Tree Regression MSE:", mse_dt_white)
print("Linear Regression RMSE:", rmse_lr_white)
print("Decision Tree Regression RMSE:", rmse_dt_white)
print(f'Linear Regression Accuracy for White Wine: {accuracy_lr_white:.2%}')
print(f'Decision Tree Accuracy for White Wine: {accuracy_dt_white:.2%}')
print("\nBest Decision Tree Model for White Wine:")
print("Best Parameters:", best_params_white)
print("Test RMSE:", rmse_best_white)

# Saving the final cleaned datasets to new CSV files
red_wine_final_file = 'cleaned_winequality-red.csv' 
white_wine_final_file = 'cleaned_winequality-white.csv'  
data_red_cleaned.to_csv(red_wine_final_file, index=False)
data_white_cleaned.to_csv(white_wine_final_file, index=False)