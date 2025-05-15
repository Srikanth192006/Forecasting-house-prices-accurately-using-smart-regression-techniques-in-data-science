import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample training data
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'bathrooms': [1, 2, 2, 3, 4],
    'location': ['A', 'B', 'A', 'C', 'B'],
    'yearBuilt': [2005, 2010, 2015, 2020, 2000],
    'price': [150000, 250000, 270000, 340000, 220000]
})

# Convert categorical location to numeric encoding
data['location_encoded'] = data['location'].apply(lambda x: hash(x) % 1000)

# Features and target
X = data[['area', 'bedrooms', 'bathrooms', 'location_encoded', 'yearBuilt']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
