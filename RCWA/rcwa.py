import numpy as np

def star_prod(A,B):
	"""
	Performs a redheffer star product for two 3D square matrices
	Assumes matrices come in the form [A_11,A_12,A_21,A_22]
	A: First list of matrices
	B: Second list of matrices
	"""

	iden_m = np.array([[1,0],
					   [0,1]])

	mid_inv_1 = B[0].dot(np.linalg.inv(iden_m-A[1].dot(B[2])))
	mid_inv_2 = A[3].dot(np.linalg.inv(iden_m-B[2].dot(A[1])))

	C = np.copy(B)

	C[0] = mid_inv_1.dot(A[0])
	C[1] = B[1]+mid_inv_1.dot(A[1].dot(B[3]))
	C[2] = A[2]+mid_inv_2.dot(B[2].dot(A[0]))
	C[3] = mid_inv_2.dot(B[3])
	
	return C