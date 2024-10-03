import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.integrate import quad

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


def main():
	"""
	All permittivities and permeabilties are relative unless stated otherwise
	"""

#################### Constants ####################

	c_0   = 299792458     # speed of light vacuum
	mu_0  = 1.257e-6      # absolute permeability vacuum
	ep_0  = 8.854e-12     # absolute permittivity vacuum
	n_inc = 1             # refractive index air

#################### Incoming Light ####################

	lambda_0 = 562e-9
	theta    = 0       # Azimuthal
	phi      = np.pi/4 # Polar
	pTE      = 0.5     # Polarization in TE
	pTM      = 1-pTE   # Polarization in TM
	w        = c_0/lambda_0 

	k_0   = w*np.sqrt(mu_0*ep_0) # Wavenumber in vacuum
#################### Device ####################

	# Reflection Region
	mu_ref = 1
	ep_ref = 1

	# Transmission Region
	mu_trm = 1
	ep_trm = 1

	# Silicon
	mu_s   = 1
	ep_s   = 15.813#-1j*0.24026

	# Grating Params 
	period = 600e-9 # [m]
	height = 400e-9 # [m]
	width  = 300e-9 # [m]
	bias   = 0

#################### Device Params ####################

	# Grid
	dim      = 512               # grid is dim x dim
	pix_size = period/dim        # pixel size
	Area     = (pix_size*dim)**2 # Area of grid

	# Creating Grating
	ER = np.ones((dim,dim))*ep_ref
	UR = np.ones((dim,dim))*mu_ref

	d = int((period-width)/pix_size/2)

	ER[:,d:-d] = ep_s
	UR[:,d:-d] = mu_s

#################### Simulation Params ####################

	# RCWA
	harmonics = 11 # At least about 10 times wavelength/period

	layers = 1

#################### Convolution Matrices ####################

	ham = int((harmonics-1)/2)
	m   = np.linspace(-ham,ham,harmonics)
	n   = np.copy(m)

	# Obtain fourier coefficients
	ERC = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ER)))
	URC = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(UR)))
	
	# Truncate coefficients at chosen harmonic
	
	place = int(dim/2)
	ERC = ERC[(place-ham):(place+1+ham),(place-ham):(place+1+ham)]
	URC = URC[(place-ham):(place+1+ham),(place-ham):(place+1+ham)]

	ERC = coeff_mat(ERC)
	URC = coeff_mat(URC)

	plt.imshow(np.real(ERC))
	plt.show()

	return


#################### Wave Vector Expansion ####################

	k_inc_scaled = n_inc*np.array([np.sin(theta)*np.cos(phi),
								   np.sin(theta)*np.sin(phi),
								   np.cos(theta)])
	
	mm,nn = np.meshgrid(m,n)

	ones = np.ones((3,3))

	T_1 = 2*np.pi*np.array([1/period,0])
	T_2 = 2*np.pi*np.array([0,1/period])

	k_x_scaled_mn = k_inc_scaled[0]*ones-mm*2*T_1[0]-nn*T_2[0]-2*np.pi/k_0/period*mm
	k_y_scaled_mn = k_inc_scaled[1]*ones-mm*2*T_1[1]-nn*T_2[1]-2*np.pi/k_0/period*nn

	k_z_ref_scaled_mn = -np.conj(np.sqrt(np.conj(mu_ref)*np.conj(ep_ref)*ones-k_x_scaled_mn@k_x_scaled_mn-k_y_scaled_mn@k_y_scaled_mn))
	k_z_trm_scaled_mn =  np.conj(np.sqrt(np.conj(mu_trm)*np.conj(ep_trm)*ones-k_x_scaled_mn@k_x_scaled_mn-k_y_scaled_mn@k_y_scaled_mn))

	K_x_scaled     = np.diag(k_x_scaled_mn.flatten())
	K_y_scaled     = np.diag(k_y_scaled_mn.flatten())
	K_z_ref_scaled = np.diag(k_z_ref_scaled_mn.flatten())
	K_z_trm_scaled = np.diag(k_z_trm_scaled_mn.flatten())

#################### Eigen-Modes Gap Medium ####################

	iden = np.identity(K_x_scaled.shape[0])
	zero = np.zeros(K_x_scaled.shape)
	
	K_z_scaled = np.conj(np.sqrt(iden-K_x_scaled@K_x_scaled-K_y_scaled@K_y_scaled))

	QTL = K_x_scaled@K_y_scaled
	QTR = iden-K_x_scaled@K_x_scaled
	QBL = K_y_scaled@K_y_scaled-iden
	QBR = -K_x_scaled@K_y_scaled

	Q   = four_matrix(QTL,QTR,QBL,QBR)

	W_0 = four_matrix(iden,zero,zero,iden)

	LTL = 1j*K_z_scaled
	LAM = four_matrix(LTL,zero,zero,LTL)

	V_0 = Q@np.linalg.inv(LAM)

#################### Initialize Device Scattering Matrix ####################

	S_11 = zero
	S_12 = iden
	S_21 = iden
	S_22 = zero

#################### Generate Device Scattering Matrix ####################

	print(K_y_scaled.shape)

	return

def four_matrix(A1,A2,A3,A4):
	"""
	Takes in 4 block matrices and returns larger matrix:

	A = | A1 A2 |
	    | A3 A4 |

	"""
	AT  = np.hstack((A1,A2))
	AB  = np.hstack((A3,A4))
	A   = np.vstack((AT,AB))
	return A

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

if __name__=="__main__":
	main()