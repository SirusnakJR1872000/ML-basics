# Implementing a Linear Regression Model using Scikit-learn:

# importing all the libraries

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = pd.DataFrame({
    'area': [1200, 1500, 1800, 2000, 2500],
    'price': [300000, 350000, 400000, 420000, 500000]
})

# now we will split our data into train and test
# where x is the feature variable and y is the test variable

x = data[['area']] #2d array
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# now we will initialize and train our model
model = LinearRegression()
model.fit(x_train, y_train)

# now we will predict and evaluate the model
y_pred = model.predict(x_test)

# now lets calculate the MSE and R2 value
print("Mean Squared Error is: ", mean_squared_error(y_test, y_pred))
print("R-squared value is: ", r2_score(y_test, y_pred))