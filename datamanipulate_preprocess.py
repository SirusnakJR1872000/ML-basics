# Data Manipulation and Preprocessing:

# Given a dataset, demonstrate how to split it into training and test sets.
# Write code to normalize/standardize a dataset.
# Implement a pipeline for feature selection and model training using scikit-learn.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Example dataset (replace with your actual dataset)
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'feature3': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'target': [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
}

# convert to dataframe
df = pd.DataFrame(data)

x = data.drop('target', axis = 1)
y = data['target']

# train - test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func = f_classif, k = 2)),
    ('classifier', RandomForestClassifier(random_state = 42))
])

# train the pipeline
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

print(classification_report(y_test, y_pred))




