import numpy as np

# Example data matrix (3x2)
data = np.array([[2, 3],
                 [5,4],
                 [4,6]])

# Step 1: Center the data by subtracting the mean of each column
centered_data = data - np.mean(data, axis=0)

# Step 2: Compute the covariance matrix of the centered data
cov_matrix = np.cov(centered_data, rowvar=False)

# Step 3: Compute eigenvalues and eigenvectors from the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort the eigenvalues and eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Output results
print("Matrix:\n", data)
print("Centered Data:\n", centered_data)
print("Covariance Matrix:\n", cov_matrix)
print("Eigenvalues:\n", sorted_eigenvalues)
print("Eigenvectors:\n", sorted_eigenvectors)

# Principal component direction
print("Principal Component Direction:\n", sorted_eigenvectors[:, 0])
