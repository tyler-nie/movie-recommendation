import numpy as mp
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)
