import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Linear Regression
"""
    - Linear regression assumes a linear relationship between the features and the target. Note this still allows for non-linear 
      transformations of the features (e.g. polynomials, logs etc) but assume that the target is a linear combination (i.e. sum) of these
      potentially transformed features
    - The goal is to fit a line (or hyperplane in multiple dimensions) that minimises the distance (residuals) between the predicted values
      and the actual values
    - Used for predicting continuous values e.g. house prices

    Key Concepts:
    1. Line of Best Fit: This is the line that best captures the trend in the data. It is determined by minimizing the sum of the squared 
      differences between the observed and predicted values (least squares method).
    2. Coefficients: These are the values that multiply the input features to produce the output. In the simplest case (single feature), it 
      consists of a slope and an intercept.
    3. Predictions: For a given set of features, predictions are made using the fitted line (model).
    4. Evaluation Metrics:
       - Mean Squared Error (MSE): Measures the average of the squared errors.
    5. R-squared (R²): 
        - Represents the proportion of the variance in the dependent variable that is predictable from the independent 
         variables
        - It ranges from 0 to 1, where 1 indicates perfect prediction (i.e. all points fall on line of best fit)
        - It is calculated as:
            R² = 1 - (mean square error / variance)
    6. Adjusted R²
        - Adding more variables into the model usually increases R^2 even if those variables are uncorrelated with the target. This is
          because with more variables, the model has more flexibility to fit to the data so generally the mean square error decreases
        - Adjusted R-squared seeks to mitigate this by penalising models with many variables. It is calculated as:
            adjusted R-squared = 1 − (n-1 / n−p−1) * (1-R²)

"""

# Seed for reproducibility
np.random.seed(5)

# Generate the original data
X = np.random.normal(0, 5, 1000)
y = 1 + 5 * X + np.random.normal(0, 20, 1000)

# Visualize the relationship between X and y
plt.figure()
plt.scatter(X, y, alpha=0.5)
plt.title('Original Data: Feature vs Target')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Reshape X and y to 2D arrays
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Convert to a DataFrame
df = pd.DataFrame(data=np.hstack([X, y]), columns=['Feature', 'Target'])

# Define the feature set and target variable
X = df[['Feature']]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Initialize the Linear Regression model
linear_regression = LinearRegression()

# Train the model
linear_regression.fit(X_train, y_train)

# Make predictions
y_pred = linear_regression.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the original model evaluation
print(f"Original Model - MSE: {mse:.2f}, R²: {r2:.2f}")
print(f"Intercept: {linear_regression.intercept_}")
print(f"Coefficient: {linear_regression.coef_[0]}")

# Combine the original feature with the new uniform-distributed feature
X_train_augmented = np.hstack((X_train, np.random.randint(0, 101, size=(len(X_train), 5))))
X_test_augmented = np.hstack((X_test, np.random.randint(0, 101, size=(len(X_test), 5))))

# Refit the model with the new augmented dataset
linear_regression_uniform_augmented = LinearRegression()
linear_regression_uniform_augmented.fit(X_train_augmented, y_train)

# Make predictions with the new augmented model
y_pred_augmented = linear_regression_uniform_augmented.predict(X_test_augmented)

# Evaluate the new augmented model
mse_uniform_augmented = mean_squared_error(y_test, y_pred_augmented)
r2_uniform_augmented = r2_score(y_test, y_pred_augmented)

print(f"New Model with additional variables - MSE: {mse_uniform_augmented:.2f}, R²: {r2_uniform_augmented:.2f}")
# Notice how we got a small improvement in r-squares despite the fact the additional variables were uncorrelated with the target
