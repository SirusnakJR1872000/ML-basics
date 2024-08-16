# Handling Missing Data in a Dataset:

# lets import all the necessary libraries

import pandas as pd
import numpy as np

# Sample dataset with missing values
data = pd.DataFrame({
    'age': [25, np.nan, 30, 35, np.nan],
    'salary': [50000, 60000, np.nan, 80000, 90000]
})


# first approach would be to drop them
data_drop = data.dropna()

# now lets try filling the missing values with median
data_fillna = data.fillna(data.mean())

# with forward fill
data_ffill_na = data.fillna(method = 'ffill')

# with backward fill
data_bfill_na = data.fillna(method = ('bfill'))