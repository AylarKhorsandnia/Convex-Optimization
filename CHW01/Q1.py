# %%
import numpy as np

# initializing the matrix
n = int(input('Size of Matrix: '))
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[[i], [j]] = input('Element ' + str(i) + ', ' + str(j) + ':')

k = int(input('How many iterations: '))

mainMat = A
# calculating the QR decomposition and creating A_i  with A_i = R_(i-1) Q_(i-1) as discussed in the part 1
for i in range(k):
    q, r = np.linalg.qr(A)
    A = r @ q

A = np.floor(A * (10 ** 8)) / (10 ** 8)

# sorting the eigenvalues in descending order
Eval = np.sort(np.diag(A))[::-1]
print(Eval)

# Creat an empty array for eigenvectors
B = np.zeros((n, n))

for i in range(n):
        # (A - lambda_i * I)
        E = mainMat - (Eval[i] * np.identity(n))
        # SVD for E
        u, s, vh = np.linalg.svd(E)
        # Transposing the V
        v = np.transpose(vh)
        # Taking the last colum of V as an eigenvector for lambda_i
        B[:, i-1] = v[:, n-1]

# Printing the B matrix which includes all eigenvectors of the main matrix
print(B)
