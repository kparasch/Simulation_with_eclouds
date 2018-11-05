import sys, os
#~ BIN = os.path.expanduser("../../../") #location of of our PyFriends
#BIN = os.path.expanduser("/afs/cern.ch/work/k/kparasch/sim_workspace/") #location of of our PyFriends
BIN = os.path.expanduser("/home/kparasch/Builds/")
sys.path.append(BIN)

import PyPARIS.communication_helpers as ch
import numpy as np
from scipy.constants import c, e
import PyPARIS.share_segments as shs
import time
import pickle
import h5py



from PyHEADTAIL.particles.slicing import UniformBinSlicer



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

class Simulation(object):
    def __init__(self):
        self.N_turns = N_turns

    def init_all(self):

        self.n_slices = n_slices
    
        
        # read the optics if needed
        if optics_pickle_file is not None:
            with open(optics_pickle_file) as fid:
                optics = pickle.load(fid)
                self.n_kick_smooth = np.sum(['_kick_smooth_' in nn for nn in optics['name']])
        else:
            optics=None
            self.n_kick_smooth = n_segments

        # define the machine
        from LHC_custom import LHC
        self.machine = LHC(n_segments = n_segments, machine_configuration = machine_configuration,
                        beta_x=beta_x, beta_y=beta_y,
                        accQ_x=Q_x, accQ_y=Q_y,
                        Qp_x=Qp_x, Qp_y=Qp_y,
                        octupole_knob=octupole_knob, 
                        optics_dict=optics)
        self.n_segments = self.machine.transverse_map.n_segments
        
        # compute sigma
        inj_opt = self.machine.transverse_map.get_injection_optics()
        sigma_x_inj = np.sqrt(inj_opt['beta_x']*epsn_x/self.machine.betagamma)
        sigma_y_inj = np.sqrt(inj_opt['beta_y']*epsn_y/self.machine.betagamma)
        
        
        if optics_pickle_file is None:
            sigma_x_smooth = sigma_x_inj
            sigma_y_smooth = sigma_y_inj
        else:
            beta_x_smooth = None
            beta_y_smooth = None
            for ele in self.machine.one_turn_map:
                if ele in self.machine.transverse_map:
                    if '_kick_smooth_' in ele.name1:
                        if beta_x_smooth is None:
                            beta_x_smooth = ele.beta_x1
                            beta_y_smooth = ele.beta_y1
                        else:
                            if beta_x_smooth != ele.beta_x1 or beta_y_smooth != ele.beta_y1:
                                raise ValueError('Smooth kicks must have all the same beta')
             
            if beta_x_smooth is None:
                sigma_x_smooth = None
                sigma_y_smooth = None
            else:
                sigma_x_smooth = np.sqrt(beta_x_smooth*epsn_x/self.machine.betagamma)
                sigma_y_smooth = np.sqrt(beta_y_smooth*epsn_y/self.machine.betagamma)

        # define MP size
        nel_mp_ref_0 = init_unif_edens_dip*4*x_aper*y_aper/N_MP_ele_init_dip
        
        # prepare e-cloud
        import PyECLOUD.PyEC4PyHT as PyEC4PyHT
        
        if enable_arc_dip:
            ecloud_dip = PyEC4PyHT.Ecloud(slice_by_slice_mode=True,
                            L_ecloud=self.machine.circumference/self.n_kick_smooth*fraction_device_dip, slicer=None, 
                            Dt_ref=Dt_ref, pyecl_input_folder=pyecl_input_folder,
                            chamb_type = chamb_type,
                            x_aper=x_aper, y_aper=y_aper,
                            filename_chm=filename_chm, 
                            Dh_sc=Dh_sc_ext,
                            PyPICmode = 'ShortleyWeller_WithTelescopicGrids',
                            f_telescope = 0.3,
                            target_grid = {'x_min_target':-target_size_internal_grid_sigma*sigma_x_smooth, 'x_max_target':target_size_internal_grid_sigma*sigma_x_smooth,
                                           'y_min_target':-target_size_internal_grid_sigma*sigma_y_smooth,'y_max_target':target_size_internal_grid_sigma*sigma_y_smooth,
                                           'Dh_target':target_Dh_internal_grid_sigma*sigma_x_smooth},
                            N_nodes_discard = 5.,
                            N_min_Dh_main = 10,
                            init_unif_edens_flag=init_unif_edens_flag_dip,
                            init_unif_edens=init_unif_edens_dip, 
                            N_mp_max=N_mp_max_dip,
                            nel_mp_ref_0=nel_mp_ref_0,
                            B_multip=B_multip_dip)
                        
        if enable_arc_quad:               
            ecloud_quad = PyEC4PyHT.Ecloud(slice_by_slice_mode=True,
                            L_ecloud=self.machine.circumference/self.n_kick_smooth*fraction_device_quad, slicer=None, 
                            Dt_ref=Dt_ref, pyecl_input_folder=pyecl_input_folder,
                            chamb_type = chamb_type,
                            x_aper=x_aper, y_aper=y_aper,
                            filename_chm=filename_chm, 
                            Dh_sc=Dh_sc_ext,
                            PyPICmode = 'ShortleyWeller_WithTelescopicGrids',
                            f_telescope = 0.3,
                            target_grid = {'x_min_target':-target_size_internal_grid_sigma*sigma_x_smooth, 'x_max_target':target_size_internal_grid_sigma*sigma_x_smooth,
                                           'y_min_target':-target_size_internal_grid_sigma*sigma_y_smooth,'y_max_target':target_size_internal_grid_sigma*sigma_y_smooth,
                                           'Dh_target':target_Dh_internal_grid_sigma*sigma_x_smooth},
                            N_nodes_discard = 5.,
                            N_min_Dh_main = 10,
                            N_mp_max=N_mp_max_quad,
                            nel_mp_ref_0=nel_mp_ref_0,
                            B_multip=B_multip_quad,
                            filename_init_MP_state=filename_init_MP_state_quad)


                        
        if self.ring_of_CPUs.I_am_the_master and enable_arc_dip:
            with open('multigrid_config_dip.txt', 'w') as fid:
                fid.write(repr(ecloud_dip.spacech_ele.PyPICobj.grids))
            
            with open('multigrid_config_dip.pkl', 'w') as fid:
                pickle.dump(ecloud_dip.spacech_ele.PyPICobj.grids, fid)
                
        if self.ring_of_CPUs.I_am_the_master and enable_arc_quad:
            with open('multigrid_config_quad.txt', 'w') as fid:
                fid.write(repr(ecloud_quad.spacech_ele.PyPICobj.grids))
            
            with open('multigrid_config_quad.pkl', 'w') as fid:
                pickle.dump(ecloud_quad.spacech_ele.PyPICobj.grids, fid)       
                

        # setup transverse losses (to "protect" the ecloud)
        import PyHEADTAIL.aperture.aperture as aperture
        apt_xy = aperture.EllipticalApertureXY(x_aper=target_size_internal_grid_sigma*sigma_x_inj, 
                                               y_aper=target_size_internal_grid_sigma*sigma_y_inj)
        self.machine.one_turn_map.append(apt_xy)
        
        
        if enable_transverse_damper:
            # setup transverse damper
            from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
            damper = TransverseDamper(dampingrate_x=dampingrate_x, dampingrate_y=dampingrate_y)
            self.machine.one_turn_map.append(damper)



        # We suppose that all the object that cannot be slice parallelized are at the end of the ring
        i_end_parallel = len(self.machine.one_turn_map)-n_non_parallelizable

        # split the machine
        sharing = shs.ShareSegments(i_end_parallel, self.ring_of_CPUs.N_nodes)
        myid = self.ring_of_CPUs.myid
        i_start_part, i_end_part = sharing.my_part(myid)
        self.mypart = self.machine.one_turn_map[i_start_part:i_end_part]
        if self.ring_of_CPUs.I_am_a_worker:
            print 'I am id=%d/%d (worker) and my part is %d long'%(myid, self.ring_of_CPUs.N_nodes, len(self.mypart))
        elif self.ring_of_CPUs.I_am_the_master:
            self.non_parallel_part = self.machine.one_turn_map[i_end_parallel:]
            print 'I am id=%d/%d (master) and my part is %d long'%(myid, self.ring_of_CPUs.N_nodes, len(self.mypart))

        #install eclouds in my part
        my_new_part = []
        self.my_list_eclouds = []
        for ele in self.mypart:
            my_new_part.append(ele)
            if ele in self.machine.transverse_map:
                if optics_pickle_file is None or '_kick_smooth_' in ele.name1:
                    if enable_arc_dip:
                        ecloud_dip_new = ecloud_dip.generate_twin_ecloud_with_shared_space_charge()
                        my_new_part.append(ecloud_dip_new)
                        self.my_list_eclouds.append(ecloud_dip_new)
                    if enable_arc_quad:
                        ecloud_quad_new = ecloud_quad.generate_twin_ecloud_with_shared_space_charge()
                        my_new_part.append(ecloud_quad_new)
                        self.my_list_eclouds.append(ecloud_quad_new)
                elif '_kick_element_' in ele.name1 and enable_eclouds_at_kick_elements:
                    
                    i_in_optics = list(optics['name']).index(ele.name1)
                    kick_name = optics['name'][i_in_optics]
                    element_name = kick_name.split('_kick_element_')[-1]
                    L_curr = optics['L_interaction'][i_in_optics]
                    
                    buildup_folder = path_buildup_simulations_kick_elements.replace('!!!NAME!!!', element_name)
                    chamber_fname = '%s_chamber.mat'%(element_name)
                    
                    B_multip_curr = [0., optics['gradB'][i_in_optics]]
                    
                    x_beam_offset = optics['x'][i_in_optics]*orbit_factor
                    y_beam_offset = optics['y'][i_in_optics]*orbit_factor
                    
                    sigma_x_local = np.sqrt(optics['beta_x'][i_in_optics]*epsn_x/self.machine.betagamma)
                    sigma_y_local = np.sqrt(optics['beta_y'][i_in_optics]*epsn_y/self.machine.betagamma)
                    
                    ecloud_ele = PyEC4PyHT.Ecloud(slice_by_slice_mode=True,
                            L_ecloud=L_curr, slicer=None, 
                            Dt_ref=Dt_ref, pyecl_input_folder=pyecl_input_folder,
                            chamb_type = 'polyg',
                            x_aper=None, y_aper=None,
                            filename_chm=buildup_folder+'/'+chamber_fname, 
                            Dh_sc=Dh_sc_ext,
                            PyPICmode = 'ShortleyWeller_WithTelescopicGrids',
                            f_telescope = 0.3,
                            target_grid = {'x_min_target':-target_size_internal_grid_sigma*sigma_x_local+x_beam_offset, 'x_max_target':target_size_internal_grid_sigma*sigma_x_local+x_beam_offset,
                                           'y_min_target':-target_size_internal_grid_sigma*sigma_y_local+y_beam_offset, 'y_max_target':target_size_internal_grid_sigma*sigma_y_local+y_beam_offset,
                                           'Dh_target':target_Dh_internal_grid_sigma*sigma_y_local},
                            N_nodes_discard=5.,
                            N_min_Dh_main=10,
                            N_mp_max=N_mp_max_quad,
                            nel_mp_ref_0=nel_mp_ref_0,
                            B_multip=B_multip_curr,
                            filename_init_MP_state=buildup_folder+'/'+name_MP_state_file_kick_elements, 
                            x_beam_offset=x_beam_offset,
                            y_beam_offset=y_beam_offset)   
                            
                    my_new_part.append(ecloud_ele)
                    self.my_list_eclouds.append(ecloud_ele)                          
                
        self.mypart = my_new_part

        if footprint_mode:
            print 'Proc. %d computing maps'%myid
            # generate a bunch 
            bunch_for_map=self.machine.generate_6D_Gaussian_bunch_matched(
                        n_macroparticles=n_macroparticles_for_footprint_map, intensity=intensity, 
                        epsn_x=epsn_x, epsn_y=epsn_y, sigma_z=sigma_z)

            # Slice the bunch
            slicer_for_map = UniformBinSlicer(n_slices = n_slices, z_cuts=(-z_cut, z_cut))
            slices_list_for_map = bunch_for_map.extract_slices(slicer_for_map)
            
            
            #Track the previous part of the machine
            for ele in self.machine.one_turn_map[:i_start_part]:
                for ss in slices_list_for_map:
                    ele.track(ss)            

            # Measure optics, track and replace clouds with maps
            list_ele_type = []
            list_meas_beta_x = []
            list_meas_alpha_x = []
            list_meas_beta_y = []
            list_meas_alpha_y = []
            for ele in self.mypart:
                list_ele_type.append(str(type(ele)))
                # Measure optics
                bbb = sum(slices_list_for_map) 
                list_meas_beta_x.append(bbb.beta_Twiss_x())
                list_meas_alpha_x.append(bbb.alpha_Twiss_x())
                list_meas_beta_y.append(bbb.beta_Twiss_y())
                list_meas_alpha_y.append(bbb.alpha_Twiss_y())
                
                if ele in self.my_list_eclouds:
                    ele.track_once_and_replace_with_recorded_field_map(slices_list_for_map)
                else:
                    for ss in slices_list_for_map:
                        ele.track(ss)       
            print 'Proc. %d done with maps'%myid

            with open('measured_optics_%d.pkl'%myid, 'wb') as fid:
                pickle.dump({
                        'ele_type':list_ele_type,
                        'beta_x':list_meas_beta_x,
                        'alpha_x':list_meas_alpha_x,
                        'beta_y':list_meas_beta_y,
                        'alpha_y':list_meas_alpha_y,
                    }, fid)
            
            #remove RF
            if self.ring_of_CPUs.I_am_the_master:
                self.non_parallel_part.remove(self.machine.longitudinal_map)
                    
    def init_master(self):
        
        # Manage multi-job operation
        if footprint_mode:
            if N_turns!=N_turns_target:
                raise ValueError('In footprint mode you need to set N_turns_target=N_turns_per_run!')
        
        import Save_Load_Status as SLS
        SimSt = SLS.SimulationStatus(N_turns_per_run=N_turns, check_for_resubmit = True, N_turns_target=N_turns_target)
        SimSt.before_simulation()
        self.SimSt = SimSt

        # generate a bunch 
        if footprint_mode:
            self.bunch = self.machine.generate_6D_Gaussian_bunch_matched(
                n_macroparticles=n_macroparticles_for_footprint_track, intensity=intensity, 
                epsn_x=epsn_x, epsn_y=epsn_y, sigma_z=sigma_z)
        elif SimSt.first_run:
            self.bunch = self.machine.generate_6D_Gaussian_bunch_matched(
                            n_macroparticles=n_macroparticles, intensity=intensity, 
                            epsn_x=epsn_x, epsn_y=epsn_y, sigma_z=sigma_z)
            
            # compute initial displacements
            inj_opt = self.machine.transverse_map.get_injection_optics()
            sigma_x = np.sqrt(inj_opt['beta_x']*epsn_x/self.machine.betagamma)
            sigma_y = np.sqrt(inj_opt['beta_y']*epsn_y/self.machine.betagamma)
            x_kick = x_kick_in_sigmas*sigma_x
            y_kick = y_kick_in_sigmas*sigma_y
            
            # apply initial displacement
            if not footprint_mode:
                self.bunch.x += x_kick
                self.bunch.y += y_kick
            
            print 'Bunch initialized.'
        else:
            print 'Loading bunch from file...'
            with h5py.File('bunch_status_part%02d.h5'%(SimSt.present_simulation_part-1), 'r') as fid:
                self.bunch = self.buffer_to_piece(np.array(fid['bunch']).copy())
            print 'Bunch loaded from file.'

        # initial slicing
        self.slicer = UniformBinSlicer(n_slices = n_slices, z_cuts=(-z_cut, z_cut))

        # define a bunch monitor 
        from PyHEADTAIL.monitors.monitors import BunchMonitor
        self.bunch_monitor = BunchMonitor('bunch_evolution_%02d'%self.SimSt.present_simulation_part,
                            N_turns, {'Comment':'PyHDTL simulation'}, 
                            write_buffer_every = 3)
        
        # define a slice monitor 
        from PyHEADTAIL.monitors.monitors import SliceMonitor
        self.slice_monitor = SliceMonitor('slice_evolution_%02d'%self.SimSt.present_simulation_part,
                            N_turns, self.slicer,  {'Comment':'PyHDTL simulation'}, 
                            write_buffer_every = 3)
        
        #slice for the first turn
        slice_obj_list = self.bunch.extract_slices(self.slicer)

        pieces_to_be_treated = slice_obj_list
        
        print 'N_turns', self.N_turns
        
        if footprint_mode:
            self.recorded_particles = ParticleTrajectories(n_macroparticles_for_footprint_track, self.N_turns)

        return pieces_to_be_treated

    def init_worker(self):
        pass

    def treat_piece(self, piece):
        for ele in self.mypart: 
                ele.track(piece)

    def finalize_turn_on_master(self, pieces_treated):
        
        # re-merge bunch
        self.bunch = sum(pieces_treated)

        #finalize present turn (with non parallel part, e.g. synchrotron motion)
        for ele in self.non_parallel_part:
            ele.track(self.bunch)
            
        # save results		
        #print '%s Turn %d'%(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()), i_turn)
        self.bunch_monitor.dump(self.bunch)
        self.slice_monitor.dump(self.bunch)
        
        # prepare next turn (re-slice)
        new_pieces_to_be_treated = self.bunch.extract_slices(self.slicer)
        
        # order reset of all clouds
        orders_to_pass = ['reset_clouds']
        
        if footprint_mode:
            self.recorded_particles.dump(self.bunch)
        
        # check if simulation has to be stopped
        if not footprint_mode and self.bunch.macroparticlenumber < sim_stop_frac * n_macroparticles:
            orders_to_pass.append('stop')
            self.SimSt.check_for_resubmit = False
            print 'Stop simulation due to beam losses.'
        
        return orders_to_pass, new_pieces_to_be_treated


    def execute_orders_from_master(self, orders_from_master):
        if 'reset_clouds' in orders_from_master:
            for ec in self.my_list_eclouds: ec.finalize_and_reinitialize()


        
    def finalize_simulation(self):
        if footprint_mode:
            # Tunes

            import NAFFlib
            print 'NAFFlib spectral analysis...'
            qx_i = np.empty_like(self.recorded_particles.x_i[:,0])
            qy_i = np.empty_like(self.recorded_particles.x_i[:,0])
            for ii in range(len(qx_i)):
                qx_i[ii] = NAFFlib.get_tune(self.recorded_particles.x_i[ii] + 1j*self.recorded_particles.xp_i[ii])
                qy_i[ii] = NAFFlib.get_tune(self.recorded_particles.y_i[ii] + 1j*self.recorded_particles.yp_i[ii])
            print 'NAFFlib spectral analysis done.'

            # Save
            import h5py
            dict_beam_status = {\
            'x_init': np.squeeze(self.recorded_particles.x_i[:,0]),
            'xp_init': np.squeeze(self.recorded_particles.xp_i[:,0]),
            'y_init': np.squeeze(self.recorded_particles.y_i[:,0]),
            'yp_init': np.squeeze(self.recorded_particles.yp_i[:,0]),
            'z_init': np.squeeze(self.recorded_particles.z_i[:,0]),
            'qx_i': qx_i,
            'qy_i': qy_i,
            'x_centroid': np.mean(self.recorded_particles.x_i, axis=1),
            'y_centroid': np.mean(self.recorded_particles.y_i, axis=1)}
                
            with h5py.File('footprint.h5', 'w') as fid:
                for kk in dict_beam_status.keys():
                    fid[kk] = dict_beam_status[kk]
        else:
            #save data for multijob operation and launch new job
            import h5py
            with h5py.File('bunch_status_part%02d.h5'%(self.SimSt.present_simulation_part), 'w') as fid:
                fid['bunch'] = self.piece_to_buffer(self.bunch)
            if not self.SimSt.first_run:
                os.system('rm bunch_status_part%02d.h5'%(self.SimSt.present_simulation_part-1))
            self.SimSt.after_simulation()

        
    def piece_to_buffer(self, piece):
        buf = ch.beam_2_buffer(piece)
        return buf

    def buffer_to_piece(self, buf):
        piece = ch.buffer_2_beam(buf)
        return piece



class ParticleTrajectories(object):
    def __init__(self, n_record, n_turns):

        # prepare storage for particles coordinates
        self.x_i = np.empty((n_record, n_turns))
        self.xp_i = np.empty((n_record, n_turns))
        self.y_i = np.empty((n_record, n_turns))
        self.yp_i = np.empty((n_record, n_turns))
        self.z_i = np.empty((n_record, n_turns))
        self.i_turn = 0
        
    def dump(self, bunch):
        
        # id and momenta after track
        id_after = bunch.id
        x_after = bunch.x
        y_after = bunch.y
        z_after = bunch.z
        xp_after = bunch.xp
        yp_after = bunch.yp

        # sort id and momenta after track
        indsort = np.argsort(id_after)
        id_after = np.take(id_after, indsort)
        x_after = np.take(x_after, indsort)
        y_after = np.take(y_after, indsort)
        z_after = np.take(z_after, indsort)
        xp_after = np.take(xp_after, indsort)
        yp_after = np.take(yp_after, indsort)

        self.x_i[:,self.i_turn] = x_after
        self.xp_i[:,self.i_turn] = xp_after
        self.y_i[:,self.i_turn] = y_after
        self.yp_i[:,self.i_turn] = yp_after
        self.z_i[:,self.i_turn] = z_after    
            
        self.i_turn += 1


