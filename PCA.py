import numpy as np

# Example data matrix (3x2), as in the exercise
data = np.array([[2, 3],
                 [5,4],
                 [4,6]])

# a: Center the data
centered_data = data - np.mean(data, axis=0)

# b: Compute the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# c: Compute eigenvalues and eigenvectors from covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# d: Sort the eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Print results
print("Matrix:\n", data)
print("Centered Data:\n", centered_data)
print("Covariance Matrix:\n", cov_matrix)
print("Eigenvalues:\n", sorted_eigenvalues)
print("Eigenvectors:\n", sorted_eigenvectors)

print("Principal Component Direction:\n", sorted_eigenvectors[:, 0])
