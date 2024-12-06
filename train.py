import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import LinearRegression

# Load the BostonHousing dataset from a local CSV file
# Replace 'BostonHousing.csv' with the path to your CSV file
dataset = pd.read_csv('BostonHousing.csv')

# Assuming the last column is the target variable
X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
pickle.dump(scaler, open('scaling.pkl', 'wb'))

# Train the Linear Regression model
regression = LinearRegression()
regression.fit(X_train, y_train)

print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

# Save the trained model
pickle.dump(regression, open('regmodel.pkl', 'wb'))

# Load the model for prediction
pickled_model = pickle.load(open('regmodel.pkl', 'rb'))

# Make predictions for the first data point
# Get the first row of features from the original dataset
sample_data = dataset.iloc[0, :-1].values.reshape(1, -1)

# Scale the sample data using the saved scaler
scaled_sample = scaler.transform(sample_data)

# Predict using the loaded model
prediction = pickled_model.predict(scaled_sample)
print("Prediction for the first data point:", prediction)
