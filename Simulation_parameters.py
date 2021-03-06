from scipy.constants import c
# machine and beam settings

# n_segments needs to be None if optics_pickle_file is specified
optics_pickle_file = 'lhc2018_25cm_only_triplets_IR15_b1_optics.pkl'
n_segments = None
beta_x =  None
beta_y =  None
Q_x = None
Q_y = None

#~ optics_pickle_file = None
#~ n_segments = 2
#~ beta_x =  400.0
#~ beta_y =  400.0
#~ Q_x = 62.27
#~ Q_y = 60.295

N_turns = 1024
N_turns_target = 1024

n_non_parallelizable = 2 #rf and aperture


intensity = 1.25e+11
epsn_x = 2.5e-6
epsn_y = 2.5e-6

machine_configuration = 'LHC-collision'

sim_stop_frac = 0.9

octupole_knob = 0.
Qp_x = 0.
Qp_y = 0.

n_slices = 50 #150
n_macroparticles = n_slices*250 #700000
sigma_z = 1.2e-9/4*c

x_kick_in_sigmas = 0.1
y_kick_in_sigmas = 0.1

# transverse damper settings
enable_transverse_damper = False
dampingrate_x = 100.
dampingrate_y = 100.
if enable_transverse_damper: n_non_parallelizable += 1


# footprint settings
footprint_mode = True
#n_macroparticles_for_footprint_map = 5000000
n_macroparticles_for_footprint_map = 500000
n_macroparticles_for_footprint_track = 5000


# general e-cloud settings
Dh_sc_ext = .8e-3
target_size_internal_grid_sigma = 10.
target_Dh_internal_grid_sigma = 0.2

chamb_type = 'polyg'
x_aper = 2.300000e-02
y_aper = 1.800000e-02
filename_chm = 'LHC_chm_ver.mat'

z_cut = 2.5e-9/2*c

Dt_ref = 5e-12
pyecl_input_folder = './pyecloud_config'


# dedicated dipole e-cloud settings
enable_arc_dip = False
fraction_device_dip = 0.65
init_unif_edens_flag_dip = 1
init_unif_edens_dip = 1.000000e+12
N_MP_ele_init_dip = 500000
N_mp_max_dip = N_MP_ele_init_dip*4
B_multip_dip = [8.33] #T

# dedicated quadrupole e-cloud settings
enable_arc_quad = False
fraction_device_quad = 7.000000e-02
N_mp_max_quad = 2000000 
B_multip_quad = [0., 188.2]
folder_path = '../../LHC_ecloud_distrib_quads/'
filename_state =  'not_used_enable_arc_quad_is_off'
filename_init_MP_state_quad = folder_path + filename_state

# dedicated quadrupole _kick_element_ settings
enable_eclouds_at_kick_elements = True
path_buildup_simulations_kick_elements = '/home/kparasch/workspace/Triplets/ec_headtail_triplets/simulations_PyECLOUD/!!!NAME!!!_sey1.35'
name_MP_state_file_kick_elements = 'MP_state_9.mat'
orbit_factor = 6.250000e-01

