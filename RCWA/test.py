import numpy as np
import scipy
import matplotlib.pyplot as plt

def conv_matrix_2D(A):
	"""
	Consider convolving two square matrices of dimension n x n
	A*B=D
	This function will return a matrix C such that
	Cb=d
	where b is a flattened version (of dimension n^2) of B.
	d is a flattened version of the output matrix, to obtain D:
	D=d.reshape(2*n-1,2*n-1)

	Only works for SQUARE matrices :)


	** Use numpy.split maybe!?!?!?!?  **
	"""
	
	n = A.shape[0]
	A = np.pad(A,n-1)
	
	Cs = []
	for i in range(2*n-1):
		for j in range(2*n-1):
			A1 = A[i:i+n,j:j+n]
			C1 = np.flip(A1.flatten())
			Cs.append(C1)

	C = np.vstack(Cs)
	
	return C

def coeff_mat(A):
	"""
	MATRIX MUST BE ODD SHAPED AND SQUARE
	"""
	
	N = A.shape[0]
	l =[]
	A_pad = np.pad(A,int((N-1)/2))

	for i in range(N):
		for j in range(N):
			if j==0:
				ls = np.rot90((A_pad[N-1:,i:N+i])).flatten()
			else:
				ls = np.rot90((A_pad[N-1-j:-j,i:N+i])).flatten()
			l.append(ls)

	return np.vstack(l)


N=3

A = np.array(range(N**2)).reshape((N,N))
# B = np.array(range(N**2,2*N**2,1)).reshape((N,N))
# C = np.ones((N,N))

print(coeff_mat(A))





