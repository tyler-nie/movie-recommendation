import numpy as mp
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it (creates an interaction matrix and stores the data in our data variable as a dictionary)
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))
