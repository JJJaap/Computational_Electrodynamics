import numpy as np
from scipy.linalg import expm
import matplotlib.plot as plt

def star_prod(S1,S2):
	iden_m = np.array([[1,0,0],
					   [0,1,0],
					   [0,0,1]])

	D = S1[1]*np.linalg.inv(iden_m-np.linalg.inv(S2[0])*S1[3])
	F = S2[2]*np.linalg.inv(iden_m-S1[3]*np.linalg.inv(S2[0]))

	S1[0] = S1[0]+D*S2[0]*S1[2]
	S1[1] = D*S2[1]
	S1[2] = F*S2[2]
	S1[3] = S2[3]+F*S1[3]*S2[1]

	return S1

def main():
	"""
	Assumes isotropic linear materials (for now)
	"""

######### Adjustable parameters #####

	w     = 550e-9/c    # Frequency of incoming EMR
	theta = np.pi/4 	# Incident wave angle [radians]
	phi   = 0			# Incident wave angle [radians]

	# Device params
	ep_r  = np.array([2.5,3.5,2])      # electric permitivities of materials
	mu_r  = np.array([1,1,1])          # magentic permeabilities of materials
	L     = np.array([0.25,0.75,0.89]) # Thickness of materials

	# Polarization factors
	p_TE  = 0.25               # relative magnitude of TE
	p_TM  = np.sqrt(1-p_TE**2) # fixed since we require p_TE^2+P_TM^2=1

#####################################

###################### Constants ##################

	c     = 299792458     # m/s
	n_inc = 1             # air
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
	iden_m = np.array([[1,0,0],
					   [0,1,0],
					   [0,0,1]])
	
	# Zero matrix
	zero_m = np.array([[0,0,0],
					   [0,0,0],
					   [0,0,0]])

	# Normal vector for transverse electric
	crss = np.cross(normal,k_inc_scaled)
	a_TE = np.array([0,1,0]) if theta==0 else crss/np.abs(crss) # Avoids ambiguity for normal incidence

	# Normal vector for transverse magnetic
	crss_TE  = np.cross(k_inc_scaled,a_TE)
	a_TM     = crss_TE/np.abs(crss_TE)

	# Polarization vector
	P = p_TE*a_TE+p_TM*a_TM

	# Initial global scattering matrix components
	# We expect no reflection and complete transmission
	# hence the assignments below
	S_global_11 = zero_m
	S_global_12 = iden_m
	S_global_21 = iden_m
	S_global_22 = zero_m

	# Reflection and transmission side scattering matrices
	# Reflection
	k_ref_z      = np.sqrt(mu_0*ep_0-k_inc_scaled[0]**2-k_inc_scaled[1]**2)
	Q_ref        = np.array([[k_inc_scaled[0]*k_inc_scaled[1] ,mu_0*ep_0+k_inc_scaled[1]**2],
						   [k_inc_scaled[0]**2-mu_0*ep_0    ,-k_inc_scaled[0]*k_inc_scaled[1]]])
	Omega_ref    = 1j*k_ref_z*iden_m
	V_ref        = Q_ref*np.linalg.inv(Omega_ref)
	A_ref        = iden_m+np.linalg.inv(V_g)*V_ref
	B_ref        = iden_m-np.linalg.inv(V_g)*V_ref
	
	S_ref_11 = -np.lingalg.inv(A_ref)*B_ref
	S_ref_12 = 2*np.lingalg.inv(A_ref)
	S_ref_21 = 1/2*(A_ref-B_ref*np.lingalg.inv(A_ref)*B_ref)
	S_ref_22 = B_ref*np.lingalg.inv(A_ref)

	# Since my transmission material is the same
	# as my reflection material, A and B are the same

	S_trm_11 = B_ref*np.lingalg.inv(A_ref)
	S_trm_12 = 1/2*(A_ref-B_ref*np.lingalg.inv(A_ref)*B_ref)
	S_trm_21 = 2*np.lingalg.inv(A_ref)
	S_trm_22 = -np.lingalg.inv(A_ref)*B_ref

###################################################

#################### Loop over Layers ####################

	for i in range(len(ep_r)):
		# Calculate layer parameters
		k_scaled_z = np.sqrt(mu_r[i]*ep_r[i]-k_inc_scaled[0]**2-k_inc_scaled[1]**2)
		Q_i        = np.array([[k_inc_scaled[0]*k_inc_scaled[1]    ,mu_r[i]*ep_r[i]+k_inc_scaled[1]**2],
				               [k_inc_scaled[0]**2-mu_r[i]*ep_r[i] ,-k_inc_scaled[0]*k_inc_scaled[1]]])
		Omega_i    = 1j*k_scaled_z*iden_m
		V_i        = Q_i*np.linalg.inv(Omega_i)
		
		# Calculate local scattering matrix
		X_i = expm(Omega_i*k_0*L[i])
		A_i   = iden_m+np.linalg.inv(V_i)*V_g
		B_i   = iden_m-np.linalg.inv(V_i)*V_g
		D     = A_i-X_i*B_i*np.linalg.inv(A_i)*X_i*B_i
		S_11  = np.linalg.inv(D)*(X_i*B_i*np.linalg.inv(A_i)*X_i*A_i-B_i)
		S_12  = np.linalg.inv(D)*X_i*(A_i-B_i*np.linalg.inv(A_i)*B_i)
		S_21  = S_12
		S_22  = S_11

		# Update global scattering matrix (Redheffer star product)
		S_global_11,S_global_12,S_global_21,S_global_22=star_prod([S_global_11,S_global_12,S_global_21,S_global_22],[S_11,S_12,S_21,S_22])

###########################################################

#################### Connect External ####################

	S_global_11,S_global_12,S_global_21,S_global_22 = star_prod([S_ref_11,S_ref_12,S_ref_21,S_ref_22],[S_global_11,S_global_12,S_global_21,S_global_22])
	S_global_11,S_global_12,S_global_21,S_global_22 = star_prod([S_global_11,S_global_12,S_global_21,S_global_22],[S_trm_11,S_trm_12,S_trm_21,S_trm_22])

###########################################################

################### Calculate Source ######################



	return

if __name__=="__main__":
	main()


