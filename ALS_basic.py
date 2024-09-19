import numpy as np
from scipy.sparse import csr_matrix

# rows = users, cols = items, 1 = like
data = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1],
])

# Convert data to a sparse matrix
user_item_matrix = csr_matrix(data)

# Parameters for ALS
F = 10              # Latent factors
alpha = 40          # Confidence scaling factor
lambda_reg = 0.1    # Regularization parameter
iterations = 10     # Number of iterations

num_users, num_items = user_item_matrix.shape

X = np.random.rand(num_users, F)
Y = np.random.rand(num_items, F)

# Precompute the confidence matrix C = 1 + alpha * R
Cui = user_item_matrix * alpha
Cui.data += 1  # C_ui = 1 + alpha * R_ui

# Create binary preference matrix P where P_ui = 1 if R_ui > 0
P = user_item_matrix.copy()
P.data = np.ones_like(P.data)


for iter in range(iterations):
    print(f"Iteration {iter + 1}/{iterations}")
    
    for u in range(num_users):
        user_row = user_item_matrix.getrow(u)
        indices = user_row.indices
        # Confidence values for the items
        Cu = Cui.getrow(u).toarray().flatten()
        # Preference values for the items
        Pu = P.getrow(u).toarray().flatten()
        
        # Cu - I (identity matrix)
        CuI = np.diag(Cu - 1)
        # Y^T * (Cu - I) * Y
        YTCuIY = Y.T @ CuI @ Y
        # Y^T * Cu * Pu^T
        YTCuPu = Y.T @ (Cu * Pu)
        # Solve for X[u]
        A = Y.T @ np.diag(Cu) @ Y + lambda_reg * np.eye(F)
        b = YTCuPu
        X[u] = np.linalg.solve(A, b)
    
    # Update item factors Y
    for i in range(num_items):
        item_col = user_item_matrix.getcol(i)
        indices = item_col.indices
        # Confidence values for the users
        Ci = Cui[:, i].toarray().flatten()
        # Preference values for the users
        Pi = P[:, i].toarray().flatten()
        
        # Ci - I (identity matrix)
        CiI = np.diag(Ci - 1)
        # X^T * (Ci - I) * X
        XTCiIX = X.T @ CiI @ X
        # X^T * Ci * Pi^T
        XTCiPi = X.T @ (Ci * Pi)
        # Y[i]
        A = X.T @ np.diag(Ci) @ X + lambda_reg * np.eye(F)
        b = XTCiPi
        Y[i] = np.linalg.solve(A, b)


user_ids = np.arange(num_users)

# Number of recommendations
N = 2

for u in user_ids:

    user_scores = X[u] @ Y.T
    # Get items the user has already interacted with
    user_interactions = user_item_matrix.getrow(u).indices
    # Exclude items the user has already interacted with
    user_scores[user_interactions] = -np.inf

    top_items = np.argsort(-user_scores)[:N]

    print(f"Recommendations for User {u + 1}: {top_items + 1}")
