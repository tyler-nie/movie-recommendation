import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it (creates an interaction matrix and stores the data in our data variable as a dictionary)
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model (using warp (Weighted Approximate-Rank Pairwise) which uses the gradient descent algorithm)
model = LightFM(loss='warp')

# Train model
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendations for each user we input
    for user_id in user_ids:

        # Movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # Get the list of positive ratings in compressed sparse row format

        # Movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # Rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print("Users %s" % user_id)
        print("      Known positives:")
        # Print the top 3 known positives
        for x in known_positives[: 3]:
            print("           %s" % x)

        print("     Recommended:")

        for x in top_items[: 3]:
            print("           %s" % x)
        # Print the top 3 recommended movies that our model predicts


sample_recommendation(model, data, [0, 214, 942, 523])
# Use as many users as you want however the user_id range 0 to 942
