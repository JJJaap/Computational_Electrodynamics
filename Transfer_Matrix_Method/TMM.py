import numpy as np
from scipy.linalg import expm

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

def main(verbose=False):
	"""
	Assumes isotropic linear materials (for now)
	"""

############# Adjustable parameters #############

	w     = 299792458/650e-9/2/np.pi     # Angular frequency of incoming EMR
	theta = np.pi/4 	                 # Incident wave polar angle [radians]
	phi   = 0			                 # Incident wave azimuthal angle [radians]

	# Device params
	ep_r  = np.array([2.5,3.5,2])      # electric permitivities of materials
	mu_r  = np.array([1,1,1])          # magnetic permeabilities of materials
	L     = np.array([0.25,0.75,0.89])*10**(-6) # Thickness of materials [m]

###################################################

###################### Constants ##################

	epsilon = 10e-6       # defines how accurate transmittance and reflectance should be to conform to energy conservation 

	c     = 299792458     # speed of light in vacuum [m/s]
	n_inc = 1             # refractive index air
	mu_0  = 1.257e-6      # permeability vacuum
	ep_0  = 8.854e-12     # permittivity vacuum

	k_0   = w*np.sqrt(mu_0*ep_0) # Wavenumber in vacuum

	# Incident wave vector
	k_inc_scaled = n_inc*np.array([np.sin(theta)*np.cos(phi),
								   np.sin(theta)*np.sin(phi),
								   np.cos(theta)])
	
	Q_g = np.array([[k_inc_scaled[0]*k_inc_scaled[1],1+k_inc_scaled[1]**2],
				    [-(1+k_inc_scaled[0]**2)        ,-k_inc_scaled[0]*k_inc_scaled[1]]])

	V_g = -1j*Q_g

	# Optical axis
	normal = np.array([0,0,1])

	# Identity matrix
	iden_m = np.array([[1,0],
					   [0,1]])
	
	# Zero matrix
	zero_m = np.array([[0,0],
					   [0,0]])

	# Initial global scattering matrix components
	# We expect no reflection and complete transmission
	# hence the assignments below
	S_global_11 = zero_m
	S_global_12 = iden_m
	S_global_21 = iden_m
	S_global_22 = zero_m

	# Reflection and transmission side scattering matrices
	# Reflection

	k_ref_z      = np.emath.sqrt(mu_0*ep_0-k_inc_scaled[0]**2-k_inc_scaled[1]**2)
	Q_ref        = np.array([[k_inc_scaled[0]*k_inc_scaled[1] ,mu_0*ep_0+k_inc_scaled[1]**2],
						   [k_inc_scaled[0]**2-mu_0*ep_0    ,-k_inc_scaled[0]*k_inc_scaled[1]]])
	Omega_ref    = 1j*k_ref_z*iden_m
	V_ref        = Q_ref.dot(np.linalg.inv(Omega_ref))
	A_ref        = iden_m+np.linalg.inv(V_g).dot(V_ref)
	B_ref        = iden_m-np.linalg.inv(V_g).dot(V_ref)
	
	S_ref_11 = -np.linalg.inv(A_ref).dot(B_ref)
	S_ref_12 = 2*np.linalg.inv(A_ref)
	S_ref_21 = 1/2*(A_ref-B_ref.dot(np.linalg.inv(A_ref).dot(B_ref)))
	S_ref_22 = B_ref.dot(np.linalg.inv(A_ref))

	# Since my transmission material is the same
	# as my reflection material, A and B are the same

	S_trm_11 = B_ref.dot(np.linalg.inv(A_ref))
	S_trm_12 = 1/2*(A_ref-B_ref.dot(np.linalg.inv(A_ref).dot(B_ref)))
	S_trm_21 = 2*np.linalg.inv(A_ref)
	S_trm_22 = -np.linalg.inv(A_ref).dot(B_ref)

##########################################################

#################### Loop over Layers ####################

	for i in range(len(ep_r)):
		# Calculate layer parameters
		k_scaled_z = np.sqrt(mu_r[i]*ep_r[i]-k_inc_scaled[0]**2-k_inc_scaled[1]**2)

		Q_i        = 1/mu_r[i]*np.array([[k_inc_scaled[0]*k_inc_scaled[1]    ,mu_r[i]*ep_r[i]+k_inc_scaled[1]**2],
				               			 [k_inc_scaled[0]**2-mu_r[i]*ep_r[i] ,-k_inc_scaled[0]*k_inc_scaled[1]]])

		Omega_i    = 1j*k_scaled_z*iden_m

		# This is the eigenvector matrix for the magnetic field components
		# Recall that the eigenvector matrix for the electric field (W) is the identity matrix
		V_i   = Q_i.dot(np.linalg.inv(Omega_i))

		# Calculate local scattering matrix
		X_i   = expm(Omega_i*k_0*L[i])
		A_i   = iden_m+np.linalg.inv(V_i).dot(V_g)
		B_i   = iden_m-np.linalg.inv(V_i).dot(V_g)
		D     = A_i-X_i.dot(B_i.dot(np.linalg.inv(A_i).dot(X_i.dot(B_i))))

		S_11  = np.linalg.inv(D).dot((X_i.dot(B_i.dot(np.linalg.inv(A_i).dot(X_i.dot(A_i)))))-B_i)
		S_12  = np.linalg.inv(D).dot(X_i.dot(A_i-B_i.dot(np.linalg.inv(A_i).dot(B_i))))
		S_21  = S_12
		S_22  = S_11


		# Update global scattering matrix (Redheffer star product)
		S = star_prod([S_global_11,S_global_12,S_global_21,S_global_22],[S_11,S_12,S_21,S_22])

		if verbose:
			if np.array_equal(S_global_11,S[0]):
				print(f"(!) No change in S_11 for layer {i}")
			if np.array_equal(S_global_12,S[1]):
				print(f"(!) No change in S_12 for layer {i}")
			if np.array_equal(S_global_21,S[2]):
				print(f"(!) No change in S_21 for layer {i}")
			if np.array_equal(S_global_22,S[3]):
				print(f"(!) No change in S_22 for layer {i}")
			print("")

		S_global_11 = S[0]
		S_global_12 = S[1]
		S_global_21 = S[2]
		S_global_22 = S[3]


################################################################

#################### Connect External Media ####################

	S_global_11,S_global_12,S_global_21,S_global_22 = star_prod([S_ref_11,S_ref_12,S_ref_21,S_ref_22],[S_global_11,S_global_12,S_global_21,S_global_22])
	S_global_11,S_global_12,S_global_21,S_global_22 = star_prod([S_global_11,S_global_12,S_global_21,S_global_22],[S_trm_11,S_trm_12,S_trm_21,S_trm_22])

################################################################

##################### Calculate Source #########################

	# Polarization factors
	p_TE  = 1/np.sqrt(2)        # relative magnitude of TE
	p_TM  = np.sqrt(1-p_TE**2)  # fixed since we require p_TE^2+P_TM^2=1

	# Normal vector for transverse electric
	crss = np.cross(normal,k_inc_scaled)
	a_TE = np.array([0,1,0]) if theta==0 else crss/np.sqrt(crss.dot(crss)) # Avoids ambiguity for normal incidence

	# Normal vector for transverse magnetic
	crss_TE  = np.cross(k_inc_scaled,a_TE)
	a_TM     = crss_TE/np.sqrt(crss_TE.dot(crss_TE))

	# Polarization vector
	P = p_TE*a_TE+p_TM*a_TM

	e_src = np.array([P[0],P[1]])

##################################################################

################### Transmitted & Reflected ######################

	e_ref = np.dot(S_global_11,e_src)
	e_trm = np.dot(S_global_21,e_src)

##################################################################

################### Longitudinal Field Components ######################

	E_ref_z = -(k_inc_scaled[0]*e_ref[0]+k_inc_scaled[1]*e_ref[1])/(k_ref_z)
	E_trm_z = -(k_inc_scaled[0]*e_trm[0]+k_inc_scaled[1]*e_trm[1])/(-k_ref_z)

#########################################################################

################### Transmittance and Reflectance #######################

	R = np.linalg.norm(np.append(e_ref,E_ref_z))**2
	T = np.linalg.norm(np.append(e_trm,E_trm_z))**2*np.real(-k_ref_z/mu_0)/np.real(k_inc_scaled[2]/mu_0)

#########################################################################

	if 1-(R**2+T**2)>epsilon:
		print("\n(-) Reflected (R) and transmitted (T) power do not conform to conservation of energy!")
		print(f"(-) {R**2:.3f}+{T**2:.3f} != 1")
		print("(-) Please review input parameters and/or code\n")
	return R,T

if __name__=="__main__":
	print(main(verbose=True))


