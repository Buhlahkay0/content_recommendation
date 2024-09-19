import numpy as np

# Rows = users, cols = items, 0 = missing rating
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [0, 1, 5, 4],
])

num_users, num_items = ratings.shape

# Compute the mean rating for each user's valid ratings
user_means = np.true_divide(ratings.sum(axis=1), (ratings != 0).sum(axis=1))

# Replace unrated items with mean rating
ratings_filled = ratings.copy()
for i in range(num_users):
    ratings_filled[i, ratings_filled[i] == 0] = user_means[i]

# Apply SVD
U, sigma_values, Vt = np.linalg.svd(ratings_filled, full_matrices=False)

# Convert the vector of singular values into a diagonal matrix
sigma_matrix = np.diag(sigma_values)

# Reduce the number of latent factors
k = 2
U_k = U[:, :k]
sigma_k = sigma_matrix[:k, :k]
Vt_k = Vt[:k, :]

# Reconstruct rating matrix
ratings_pred = np.dot(np.dot(U_k, sigma_k), Vt_k)

# Generate Recommendations
N = 2

for user_index in range(num_users):
    user_ratings = ratings_pred[user_index]
    rated_items = ratings[user_index] != 0
    # Exclude already rated items from recommendation
    user_ratings[rated_items] = -np.inf  # Assign negative infinity to rated items
    top_items = np.argsort(-user_ratings)[:N]

    print(f"Recommendations for User {user_index + 1}: {top_items + 1}")
