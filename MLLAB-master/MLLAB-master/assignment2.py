import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

# Creating and training the model
model = LinearRegression().fit(x, y)

# Model parameters
r_sq = model.score(x, y)  # R-squared value
intercept = model.intercept_
slope = model.coef_

# Predictions
y_pred = model.predict(x)

# Evaluation Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Printing results
print(f"Coefficient of Determination (R^2): {r_sq}")
print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Plotting the results
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', linestyle='--', label='Regression line')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Regression Model')
plt.legend()
plt.show()