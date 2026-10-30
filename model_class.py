
###################################################################
###################################################################
## 
## 
## Large Scale Model of a Maccaque Cortex
## 
## 
## 
## 


###################################################################
###################################################################
## IMPORT SECTION 
## 
## 

from tqdm.notebook import tqdm
import numpy as np
import brian2 as br
from IPython.display import Math, Latex
from sympy import latex
from timer import Timer

pA = br.pA
nA = br.nA
Hz = br.Hz
second = br.second

import plotly as plty
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


notebook_relative_location = "../"


###################################################################
###################################################################
## CLASS
## 


class Model:

    n_area: int
    hier_vals: np.ndarray
    fln_m: np.ndarray
    fb: np.ndarray
    t_plot: np.ndarray
    n_t: int
    xi_1: np.ndarray
    xi_2: np.ndarray
    xi_i: np.ndarray
    i_ext: np.ndarray
    time_idx: np.ndarray
    area_idx: np.ndarray
    s_1n: np.ndarray
    s_2n: np.ndarray
    s_1a: np.ndarray
    s_2a: np.ndarray
    s_i: np.ndarray
    i_1: np.ndarray
    i_2: np.ndarray
    i_i: np.ndarray
    r_1: np.ndarray
    r_2: np.ndarray
    r_i: np.ndarray
    noise_1: np.ndarray
    noise_2: np.ndarray
    noise_i: np.ndarray
    lr_spinecount: np.ndarray
    w_e_n: np.ndarray
    w_e_a: np.ndarray
    w_i_n: np.ndarray
    w_i_a: np.ndarray
    j_ns: np.ndarray
    j_nc: np.ndarray
    j_as: np.ndarray
    j_ac: np.ndarray
    r_1_: np.ndarray
    r_2_: np.ndarray
    r_i_: np.ndarray

    def __init__(self,
                talky=False, network_params=None,timer=False,timer_keys=None):
        if network_params is None:
            network_params = {}
        self.network_params = network_params
        self.area_act = network_params['active_area']
        self.talky = talky
        self.timer = timer
        self.timer_keys = timer_keys

        self.initiate()


    #############################################################################################
    #############################################################################################
    ## INITIATE MODEL
    ##

    def initiate(self):
        self.load_anatomical_data()
        self.normalize_fln()
        self.set_time()
        self.set_noise()
        self.set_phase_dynamics()
        self.set_stimulus()
        self.set_attention_dynamics()
        self.set_containers()
        self.set_initial_state()
        self.compute_couplings()
        self.initiate_timer()

    def load_anatomical_data(self):
        ###################################################################################
        ## Load anatomical data.
        ## Adding the anatomical data to self.network_params
        ## Creating self.hier_vals : 1D numpy array 
        ## Creating self.n_area : int

        ## Extracting area number
        self.n_area = self.network_params['area_number']

        ## Loading anatomical data
        if "connectivity_file" in self.network_params.keys():
            f_name = self.network_params["connectivity_file"]

        try:
            with open('data/'+f_name, 'rb') as f_:
                anat_data = np.load(f_, encoding='bytes', allow_pickle=True)
        except FileNotFoundError:
            with open(notebook_relative_location+'data/'+f_name, 'rb') as f_:
                anat_data = np.load(f_, encoding='bytes', allow_pickle=True)


        self.network_params.update(anat_data)
        
        ## Loading hierarchy
        if "hierarchy" in self.network_params.keys():
            self.hier_vals = self.network_params['hierarchy']
        else:
            f_name = self.network_params['spine_count_file']

            try:
                with open('data/'+f_name, 'rb') as f_:
                    self.hier_vals = np.load(f_)
            except FileNotFoundError:
                with open(notebook_relative_location+'data/'+f_name, 'rb') as f_:
                    self.hier_vals = np.load(f_)

        self.hier_vals = self.hier_vals/np.max(self.hier_vals) # Normalizing hierarchy

    def normalize_fln(self):
        ###################################################################################
        ## Compress the range of the FLNs, and then renormalize so FLNs into
        ## each area sum to 1.
        ## Creating self.fln_m : 2D numpy array

        p = self.network_params

        ## Fln[i,j] : connection from area j to area i
        fln_m = p['fln_a'] * p[b'fln'] ** p['fln_b']  ## Equation (11) 
        if len(fln_m.shape) == 2:  # If not, fln_m is a number and the network has only one area
            fln_m = np.nan_to_num((fln_m.T/np.nansum(fln_m, 1)).T)  # Normalization needed after custom scaling of fln

        ## Creating class vars
        if len(fln_m.shape) == 2:
            self.fln_m = fln_m[:self.n_area,:self.n_area]
            self.sln = p[b'sln'][:self.n_area,:self.n_area]
            self.fb = (1-p[b'sln'])[:self.n_area,:self.n_area]
        else:
            self.fln_m = fln_m
            self.sln = p[b'sln']
            self.fb = (1-p[b'sln'])

    def set_time(self):
        ###################################################################################
        ## Define time.
        ## Creating self.t_plot : 1D numpy array
        ## Creating self.n_t : int

        p = self.network_params
        self.t_plot = np.linspace(0, p['run_t'], int(p['run_t'] / p['dt']) + 1)
        self.n_t = len(self.t_plot)

    def set_noise(self):
        ###################################################################################
        ## Define noise.
        ## Creating self.xi_1 : 1D numpy array
        ## Creating self.xi_2 : 1D numpy array
        ## Creating self.xi_i : 1D numpy array 

        self.xi_1 = np.random.randn(self.n_t, self.n_area)
        self.xi_2 = np.random.randn(self.n_t, self.n_area)
        self.xi_i = np.random.randn(self.n_t, self.n_area)

    def set_phase_dynamics(self):
        """Initialize optional regional oscillation phases for CTC experiments."""
        p = self.network_params
        self.phase = np.zeros((self.n_t, self.n_area), dtype=float)
        self.phase[0, :] = np.random.uniform(-np.pi, np.pi, self.n_area)
        self.phase_frequency = np.random.normal(
            p.get('phase_frequency_hz', 40.0),
            p.get('phase_frequency_spread_hz', 0.0),
            self.n_area,
        )
        self.phase_gate_strength = float(p.get('phase_gate_strength', 0.0))
        self.phase_coupling_rad_s = float(p.get('phase_coupling_rad_s', 0.0))
        if not 0.0 <= self.phase_gate_strength <= 1.0:
            raise ValueError("phase_gate_strength must be between 0 and 1")
        if self.phase_coupling_rad_s < 0.0:
            raise ValueError("phase_coupling_rad_s must be non-negative")

    def get_phase(self):
        """Return regional oscillator phases in radians, shape (time, areas)."""
        return self.phase

    def get_attention_gain(self):
        """Return the dimensionless attentional gain over time."""
        return self.attention_gain

    def get_attention_ignition(self):
        """Return the smooth workspace ignition fraction over time (0 to 1)."""
        return self.attention_ignition

    def get_phase_gate(self, phase):
        """Return directed source-to-target CTC gates for one phase snapshot."""
        phase_lag = float(self.network_params.get('phase_lag_rad', 0.0))
        phase_difference = phase[None, :] - phase[:, None] - phase_lag
        return 1.0 + self.phase_gate_strength * np.cos(phase_difference)

    def set_initial_state(self):
        ###################################################################################
        ## Define initial state.
        ## All values have been computed experimentally to match the steady state of the network
        
        p = self.network_params

        s_1n_init = p["s_1n_init"]
        s_2n_init = p["s_2n_init"]
        s_1a_init = p["s_1a_init"]
        s_2a_init = p["s_2a_init"]
        s_i_init = p["s_i_init"]
        i_1_init = 0 * pA
        i_2_init = 0 * pA
        i_i_init = 0 * pA
        r_1_init = p["r_init"]
        r_2_init = p["r_init"]
        r_i_init = p["ri_init"]

        self.s_1n[0, :] = s_1n_init
        self.s_1n = np.tile(self.network_params['s_1n_init'],self.n_t).reshape((self.n_t,self.n_area)) 
        self.s_2n[0, :] = s_2n_init
        self.s_2n = np.tile(self.network_params['s_2n_init'],self.n_t).reshape((self.n_t,self.n_area)) 
        self.s_1a[0, :] = s_1a_init
        self.s_1a = np.tile(self.network_params['s_1a_init'],self.n_t).reshape((self.n_t,self.n_area)) 
        self.s_2a[0, :] = s_2a_init
        self.s_2a = np.tile(self.network_params['s_2a_init'],self.n_t).reshape((self.n_t,self.n_area)) 
        self.s_i[0, :] = s_i_init
        self.i_1[0, :] = i_1_init
        self.i_2[0, :] = i_2_init
        self.i_i[0, :] = i_i_init
        self.r_1[0, :] = r_1_init
        self.r_2[0, :] = r_2_init
        self.r_i[0, :] = r_i_init

    def set_stimulus(self):
        ###################################################################################
        ## Define the stimulus.

        p = self.network_params
        self.i_ext = np.zeros((self.n_t, self.n_area)) * pA
        self.time_idx = (self.t_plot > p['stim_on']) & (self.t_plot <= p['stim_off'])


        self.sensory_indices = []
        for area in self.area_act:
            self.area_idx = p[b'area_list'].index(area)
            self.sensory_indices.append(self.area_idx)
            self.i_ext[self.time_idx, self.area_idx] = p['i_ext']

        self.vigilance = self.i_ext*0
        if "vigilance" in p.keys():
            self.vigilance += p["vigilance"]

        if "vigilance_value" in p.keys():
            vigilance = self.compute_vigilance(p['vigilance_value'])
            self.vigilance += vigilance

    def set_attention_dynamics(self):
        """Prepare task-driven gain and resolve workspace/sensory populations."""
        p = self.network_params
        self.attention_baseline_gain = float(p.get('attention_baseline_gain', 1.0))
        self.attention_task_amplitude = float(p.get('attention_task_gain', 0.0))
        self.attention_reentry_gain = float(p.get('attention_reentry_gain', 0.0))
        self.attention_reentry_current = p.get('attention_reentry_current', 0 * pA)
        self.attention_sensory_floor = p.get('attention_sensory_floor_hz', 5 * Hz)
        self.attention_ignition_threshold = p.get('attention_ignition_threshold_hz', 20 * Hz)
        self.attention_ignition_slope = p.get('attention_ignition_slope_hz', 2 * Hz)

        if self.attention_task_amplitude < 0 or self.attention_reentry_gain < 0:
            raise ValueError("Attention task and re-entry gains must be non-negative")
        if self.attention_ignition_slope <= 0 * Hz:
            raise ValueError("attention_ignition_slope_hz must be positive")

        area_names = p[b'area_list']
        workspace_areas = p.get('workspace_areas', [])
        self.workspace_indices = [area_names.index(area) for area in workspace_areas]
        if (self.attention_reentry_gain > 0 or
                self.attention_reentry_current > 0 * pA) and not self.workspace_indices:
            raise ValueError(
                "Set network_params['workspace_areas'] to enable ignition feedback"
            )

        task_on = p.get('attention_task_on', p['stim_on'])
        task_off = p.get('attention_task_off', p['stim_off'])
        task_tau = p.get('attention_task_tau', 50 * ms)
        if task_tau <= 0 * ms:
            raise ValueError("attention_task_tau must be positive")
        self.attention_task_trace = np.zeros(self.n_t, dtype=float)
        task_active = (self.t_plot >= task_on) & (self.t_plot <= task_off)
        task_decay = self.t_plot > task_off
        self.attention_task_trace[task_active] = self.attention_task_amplitude
        self.attention_task_trace[task_decay] = (
            self.attention_task_amplitude
            * np.exp(-(self.t_plot[task_decay] - task_off) / task_tau)
        )
        self.stimulus_seen = False

    def set_containers(self):
        ###################################################################################
        ## Define containers.

        self.s_1n = np.zeros((self.n_t, self.n_area))
        self.s_2n = np.zeros((self.n_t, self.n_area))
        self.s_1a = np.zeros((self.n_t, self.n_area))
        self.s_2a = np.zeros((self.n_t, self.n_area))
        self.s_i = np.zeros((self.n_t, self.n_area))
        self.i_1 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_2 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_i = np.zeros((self.n_t, self.n_area)) * pA
        self.i_loc_1 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_loc_2 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_loc_i = np.zeros((self.n_t, self.n_area)) * pA
        self.i_lr_1 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_lr_2 = np.zeros((self.n_t, self.n_area)) * pA
        self.i_lr_i = np.zeros((self.n_t, self.n_area)) * pA
        self.r_1 = np.zeros((self.n_t, self.n_area)) * Hz
        self.r_2 = np.zeros((self.n_t, self.n_area)) * Hz
        self.r_i = np.zeros((self.n_t, self.n_area)) * Hz
        self.attention_gain = np.full(
            self.n_t, self.attention_baseline_gain, dtype=float
        )
        self.attention_ignition = np.zeros(self.n_t, dtype=float)
        self.i_reentry = np.zeros((self.n_t, self.n_area)) * pA
        self.noise_1 = np.zeros((self.n_t, self.n_area)) * pA 
        self.noise_2 = np.zeros((self.n_t, self.n_area)) * pA
        self.noise_i = np.zeros((self.n_t, self.n_area)) * pA

    def compute_couplings(self):
        ###################################################################################
        ## Precompute additional variables.

        self.compute_local_couplings()

        self.compute_lr_couplings()


    def compute_local_couplings(self):

        p = self.network_params

        self.z = p['z_min'] + self.hier_vals*(1-p['z_min'])
        self.zI = p['zI_min'] + self.hier_vals*(1-p['zI_min'])

        z = self.z
        zI = self.zI
        kappa = p['kappa_loc']
        gN = p['gN']
        gA = p['gA']
        Gs = p['g_s']
        Gc = p['g_c']
        G0 = p['g_0']
        Gei = p['j_gei']
        Gie = p['g_ie'] 

        zeta = self.zeta = self.compute_zeta(p['tau_g'],p['gamma_g'],p['c_1'],p['g_i'],p['j_gii']) 
        
        self.j_ns = z * kappa * gN * Gs # Equation (7) 
        self.j_as = z * (1-kappa) * gA * Gs # Equation (8)
        self.j_nc = 0*pA
        self.j_ac = 0*pA



        self.j_nie = zI * Gie * gN # Equation (10)
        self.j_aie = 0*pA


    def compute_lr_couplings(self):

        p = self.network_params
        fln = self.fln_m
        sln = self.sln
        k_ff = p['nu_ff_n']
        k_fb = p['nu_fb_n']
        nu_ff = p['nu_ff_e']
        nu_fb = p['nu_fb_e']
        G = p['G']
        Gi = p['Gi']
        gN = p['gN']
        gA = p['gA']
        z = self.z
        zI = self.zI

        self.w_e_n_int = w_e_n_int = fln * ( sln * k_ff * nu_ff + (1-sln) * k_fb * nu_fb )
        w_e_a_int = fln * ( sln * (1-k_ff) * nu_ff + (1-sln) * (1-k_fb) * nu_fb ) 
        w_i_n_int = fln * ( sln * k_ff * (1-nu_ff) + (1-sln) * k_fb * (1-nu_fb) )
        w_i_a_int = fln * ( sln * (1-k_ff) * (1-nu_ff) + (1-sln) * (1-k_fb) * (1-nu_fb) ) 

        
        self.w_e_n = G * gN * ( z * w_e_n_int.T).T # Equation (16)
        self.w_e_a = G * gA * ( z * w_e_a_int.T).T # Equation (18)
        self.w_i_n = Gi * gN *( zI * w_i_n_int.T).T # Equation (19)
        self.w_i_a = Gi * gA *( zI * w_i_a_int.T).T # Equation (20)


    def initiate_timer(self):
        if self.timer:
            self.timer_object = Timer(self.timer_keys,self.n_t)
            

    #############################################################################################
    #############################################################################################
    ## RUN MODEL
    ##
    def _excitatory_f_i_curve(self,
            i: np.array,
            i_t: float):
            p = self.network_params
            return self.f(i[i_t, :], p['a_e'], p['b_e'], p['d_e'])

    def _inhibitory_f_i_curve(self,
            i: np.array,
            i_t: float):
            p = self.network_params
            return self.f_i(i[i_t, :], p['g_i'], p['c_1'], p['c_0'], p['r_0'])



    def run(self):
        p = self.network_params
        fixed_vars = p['fixed_vars']
        s_1n,s_2n,s_1a,s_2a,s_i = self.s_1n,self.s_2n,self.s_1a,self.s_2a,self.s_i
        r_1,r_2,r_i = self.r_1,self.r_2,self.r_i

        tau_n,gamma_n,tau_a,gamma_a,tau_g,gamma_g,tau_r,dt = p['tau_n'],p['gamma_n'],p['tau_a'],p['gamma_a'],p['tau_g'],p['gamma_g'],p['tau_r'],p['dt']
        phase_dt_s = float(dt / second)

        noise_1, noise_2, noise_i = self.noise_1,self.noise_2,self.noise_i

        sigma_e,sigma_i = p['sigma_e'],p['sigma_i']

        xi_1,xi_2,xi_i = self.xi_1, self.xi_2, self.xi_i

        j_ns,j_as,j_nc,j_ac =  self.j_ns,self.j_as,self.j_nc,self.j_ac

        j_gei,j_gii = p['j_gei'],p['j_gii']

        j_nie,j_aie = self.j_nie,self.j_aie


        w_e_n,w_e_a,w_i_n,w_i_a = self.w_e_n,self.w_e_a,self.w_i_n,self.w_i_a

        custom_range = range(1, self.n_t)
        if self.talky:
            custom_range = tqdm(custom_range)

        for i_t in custom_range:

            # Update the smooth ignition signal from the previous time step to
            # avoid an algebraic loop in this explicit Euler model.
            if self.workspace_indices:
                workspace_rate_hz = float(
                    np.mean(r_1[i_t - 1, self.workspace_indices]) / Hz
                )
                threshold_hz = float(self.attention_ignition_threshold / Hz)
                slope_hz = float(self.attention_ignition_slope / Hz)
                logistic_input = np.clip(
                    (workspace_rate_hz - threshold_hz) / slope_hz, -60, 60
                )
                ignition_fraction = 1.0 / (1.0 + np.exp(-logistic_input))
            else:
                ignition_fraction = 0.0
            self.attention_ignition[i_t] = ignition_fraction
            self.attention_gain[i_t] = (
                self.attention_baseline_gain
                + self.attention_task_trace[i_t]
                + self.attention_reentry_gain * ignition_fraction
            )

            # Re-entry is enabled only after a sensory stimulus has occurred
            # and the target sensory population remains active.
            if np.any(self.i_ext[i_t, self.sensory_indices] > 0 * pA):
                self.stimulus_seen = True
            if self.stimulus_seen and self.attention_reentry_current > 0 * pA:
                for sensory_idx in self.sensory_indices:
                    if r_1[i_t - 1, sensory_idx] >= self.attention_sensory_floor:
                        self.i_reentry[i_t, sensory_idx] = (
                            self.attention_reentry_current * ignition_fraction
                        )

            # Kuramoto-style phase dynamics. The anatomical FLN matrix supplies
            # the directed inter-areal coupling weights.
            phase_previous = self.phase[i_t - 1, :]
            if self.phase_coupling_rad_s == 0.0:
                phase_interaction = 0.0
            else:
                phase_difference = phase_previous[None, :] - phase_previous[:, None]
                phase_interaction = np.sum(
                    self.fln_m * np.sin(phase_difference), axis=1
                )
            self.phase[i_t, :] = np.mod(
                phase_previous + phase_dt_s * (
                    2 * np.pi * self.phase_frequency
                    + self.phase_coupling_rad_s * phase_interaction
                ),
                2 * np.pi,
            )


            if self.timer:
                self.timer_object.initiate('s')

            ## Update sn
            d_s1n = dt * (-s_1n[i_t - 1, :] / tau_n + gamma_n * (1 - s_1n[i_t - 1, : ]) * r_1[i_t - 1, :]) # Equation (3)
            s_1n[i_t, :] = s_1n[i_t - 1, :] + d_s1n

            d_s2n = dt * (-s_2n[i_t - 1, :] / tau_n + gamma_n * (1 - s_2n[i_t - 1, : ]) * r_2[i_t - 1, :]) # Equation (3)
            s_2n[i_t, :] = s_2n[i_t - 1, :] + d_s2n


            ## Update sa
            d_s1a = dt * (-s_1a[i_t - 1, :] / tau_a + gamma_a * (1 - s_1a[i_t - 1, : ]) * r_1[i_t - 1, :]) # Equation (4)
            s_1a[i_t, :] = s_1a[i_t - 1, :] + d_s1a

            d_s2a = dt * (-s_2a[i_t - 1, :] / tau_a + gamma_a * (1 - s_2a[i_t - 1, : ]) * r_2[i_t - 1, :]) # Equation (4)
            s_2a[i_t, :] = s_2a[i_t - 1, :] + d_s2a

            ## Update si
            d_si = dt * (-s_i[i_t - 1, : ] / tau_g + gamma_g * r_i[i_t - 1, :]) # Equation (5) (Mismatch)
            s_i[i_t, :] = s_i[i_t - 1, :] + d_si


            if self.timer:
                self.timer_object.add_time('s',i_t)


            if self.timer:
                self.timer_object.initiate('noise')
 
            ## Update noise
            d_noise1 = dt * (-noise_1[i_t - 1, :] / tau_a + xi_1[i_t, :] * sigma_e / np.sqrt(dt * tau_a) ) # Equation (14) (Mismatch)
            noise_1[i_t, :] = noise_1[i_t - 1, : ] + d_noise1

            d_noise2 = dt * (-noise_2[i_t - 1, :] / tau_a + xi_2[i_t, :] * sigma_e / np.sqrt(dt * tau_a) ) # Equation (14) (Mismatch)
            noise_2[i_t, :] = noise_2[i_t - 1, : ] + d_noise2

            d_noisei = dt * (-noise_i[i_t - 1, :] / tau_a + xi_i[i_t, :] * sigma_i / np.sqrt(dt * tau_a) ) # Equation (14) (Mismatch)
            noise_i[i_t, :] = noise_i[i_t - 1, : ] + d_noisei

            if self.timer:
                self.timer_object.add_time('noise',i_t)

            if self.timer:
                self.timer_object.initiate('local_current')


            ## Update local currents
            self.i_loc_1[i_t, :] = (j_ns * s_1n[i_t, :] + j_nc * s_2n[i_t, :] + # Equation (7)
                    j_as * s_1a[i_t, :] + j_ac * s_2a[i_t, :] + # Equation (8)
                    j_gei * s_i[i_t, : ]) # Equation (9)

            self.i_loc_2[i_t, :] = (j_ns * s_2n[i_t, :] + j_nc * s_1n[i_t, :] + # Equation (7)
                    j_as * s_2a[i_t, :] + j_ac * s_1a[i_t, :] + # Equation (8)
                    j_gei * s_i[i_t, : ]) # Equation (9)

            self.i_loc_i[i_t, :] = (j_nie * s_1n[i_t, :] + j_nie * s_2n[i_t, :] + # Equation (7)
                    j_aie * s_1a[i_t, :] + j_aie * s_2a[i_t, :] + # Equation (8)
                    j_gii * s_i[i_t, : ]) # Equation (9)


            if self.timer:
                self.timer_object.add_time('local_current',i_t)

            if self.timer:
                self.timer_object.initiate('lr_current')

            ## Update longrange currents
            if self.phase_gate_strength == 0.0:
                self.i_lr_1[i_t, :] = np.clip((np.dot(s_1n[i_t, :], w_e_n.T) + np.dot(s_1a[i_t, :], w_e_a.T)), 0*pA, 300*pA) # Equation (16) & (17) & (18)
                self.i_lr_2[i_t, :] = np.clip((np.dot(s_2n[i_t, :], w_e_n.T) + np.dot(s_2a[i_t, :], w_e_a.T)), 0*pA, 300*pA) # Equation (16) & (17) & (18)
                self.i_lr_i[i_t, :] = np.clip((np.dot(s_1n[i_t, :] + s_2n[i_t, :], w_i_n.T) + np.dot(s_1a[i_t, :]+s_2a[i_t, :], w_i_a.T)), 0*pA, 300*pA) # Equation (20) & (19)
            else:
                phase_gate = self.get_phase_gate(phase_previous)
                self.i_lr_1[i_t, :] = np.clip(
                    np.sum(w_e_n * phase_gate * s_1n[i_t, :][None, :], axis=1)
                    + np.sum(w_e_a * phase_gate * s_1a[i_t, :][None, :], axis=1),
                    0*pA, 300*pA,
                )
                self.i_lr_2[i_t, :] = np.clip(
                    np.sum(w_e_n * phase_gate * s_2n[i_t, :][None, :], axis=1)
                    + np.sum(w_e_a * phase_gate * s_2a[i_t, :][None, :], axis=1),
                    0*pA, 300*pA,
                )
                self.i_lr_i[i_t, :] = np.clip(
                    np.sum(w_i_n * phase_gate * (s_1n[i_t, :] + s_2n[i_t, :])[None, :], axis=1)
                    + np.sum(w_i_a * phase_gate * (s_1a[i_t, :] + s_2a[i_t, :])[None, :], axis=1),
                    0*pA, 300*pA,
                )
            

            if self.timer:
                self.timer_object.add_time('lr_current',i_t)

            if self.timer:
                self.timer_object.initiate('current')

            # Update full currents
            self.i_1[i_t, :] = (
                self.i_loc_1[i_t, :] + self.i_lr_1[i_t, :] + p['i_bg_e']
                + self.noise_1[i_t, :] + self.vigilance[i_t, :]
                + self.i_ext[i_t, :] * self.attention_gain[i_t]
                + self.i_reentry[i_t, :]
            )
            self.i_2[i_t, :] = self.i_loc_2[i_t, :] + self.i_lr_2[i_t, :] + p['i_bg_e'] + self.noise_2[i_t, :] + self.vigilance[i_t, :] 
            self.i_i[i_t, :] = self.i_loc_i[i_t, :] + self.i_lr_i[i_t, :] + p['i_bg_i'] + self.noise_i[i_t, :] 

            if self.timer:
                self.timer_object.add_time('current',i_t)

            if self.timer:
                self.timer_object.initiate('fr')

        
            ## Update firing rates
            r1_ = self._excitatory_f_i_curve(self.i_1,i_t)
            dr1 = dt * (-r_1[i_t - 1, :] / tau_r + r1_ / tau_r)
            r_1[i_t, :] = r_1[i_t - 1, :] + dr1

            r2_ = self._excitatory_f_i_curve(self.i_2,i_t)
            dr2 = dt * (-r_2[i_t - 1, :] / tau_r + r2_ / tau_r)
            r_2[i_t, :] = r_2[i_t - 1, :] + dr2

            ri_ = self._inhibitory_f_i_curve(self.i_i,i_t)
            dri = dt * (-r_i[i_t - 1, :] / tau_r + ri_ / tau_r)
            r_i[i_t, :] = r_i[i_t - 1, :] + dri




    #############################################################################################
    #############################################################################################
    ## COMPUTATIONS FUNCTIONS
    ##

    def compute_inhibitory_hierarchy(self):


        p = self.network_params

        self.zeta_2 = (2 * p['c_1'] * p['j_gei'] * p['tau_g'] * p['gamma_g'] /
                  (p['c_1'] * p['j_gii'] * p['tau_g'] * p['gamma_g'] - p['g_i']))


        self.j_nie_grad = 1 / (2 * p['j_gei'] * self.zeta) * (211.2*pA - (210+self.hier_vals*(440-210))*pA -
                                                 10.7*pA)/ self.zeta_2

        self.j_nie_grad /= self.j_nie_grad.max()


    #############################################################################################
    #############################################################################################
    ## STATIC FUNCTIONS 
    ##

    @staticmethod
    def f(i, a=.27*Hz/pA, b=108*Hz, d=.308*second):
        return (a * i - b) / (1 - np.exp(-d * (a * i - b))) ## Equation (19)

    @staticmethod
    def f_i(i, g_i=4., c_1=.615*Hz/pA, c_0=177*Hz, r_0=5.5*Hz):
        rate = 1 / g_i * (c_1 * i - c_0) + r_0 ## Equation (20)
        return rate * (rate > 0)

    @staticmethod
    def compute_zeta(tau_g,gamma_1,c_1,g_I,JII):
        return (tau_g*gamma_1*c_1)/(g_I+np.abs(JII)*tau_g*gamma_1*c_1)


    @staticmethod
    def compute_beta(tau_g,gamma_1,c_1,g_I,JII,ibgi,c0,r0):
        return (tau_g*gamma_1)*(c_1*ibgi+g_I*r0-c0)/(g_I-np.abs(JII)*tau_g*gamma_1*c_1)

    #############################################################################################
    #############################################################################################
    ## GETTING FUNCTIONS 
    ##

    def get_r(self, i=1):
        if i == 1:
            return self.r_1
        else:
            return self.r_2

    def get_ri(self):
        return self.r_i

    def get_sn(self, i=1):
        if i == 1:
            return self.s_1n
        else:
            return self.s_2n

    def get_sa(self, i=1):
        if i == 1:
            return self.s_1a
        else:
            return self.s_2a

    def get_si(self):
        return self.s_i

    def get_i(self, i=1):
        if i == 1:
            return self.i_1
        elif i == 2:
            return self.i_2
        elif i == 'i':
            return self.network_params['j_gei']*self.s_i
        else:
            return self.i_ext

    def get_spec_i(self, exc=True, nmda=True, i=1):
        p = self.network_params
        if exc and nmda and i == 1:
            return np.dot(self.s_1n, self.w_e_n.T)
        elif exc and (not nmda) and i == 1:
            return np.dot(self.s_1a, self.w_e_a.T)
        elif exc and nmda and i == 0:
            return np.dot(self.s_2n, self.w_e_n.T)
        elif exc and (not nmda) and i == 0:
            return np.dot(self.s_2a, self.w_e_a.T)
        elif (not exc) and nmda:
            return np.dot(self.s_1n + self.s_2n, self.w_i_n.T)
        elif (not exc) and (not nmda):
            return np.dot(self.s_1a + self.s_2a, self.w_i_a.T)

    def get_local_i(self, rec=True, nmda=True, i=1):
        if rec and nmda and i == 1:
            return self.j_ns * self.s_1n
        elif rec and (not nmda) and i == 1:
            return self.j_as * self.s_1a
        elif rec and nmda and i == 0:
            return self.j_ns * self.s_2n
        elif rec and (not nmda) and i == 0:
            return self.j_as * self.s_2a
        elif (not rec) and nmda and i == 1:
            return self.j_nc * self.s_2n
        elif (not rec) and (not nmda) and i == 1:
            return self.j_ac * self.s_2a
        elif (not rec) and nmda and i == 0:
            return self.j_nc * self.s_1n
        elif (not rec) and (not nmda) and i == 0:
            return self.j_ac * self.s_1a

    def get_ii(self):
        return self.i_i

    def get_noise(self, i=1):
        if i == 1:
            return self.noise_1
        else:
            return self.noise_2

    def get_noisei(self):
        return self.noise_i

    #############################################################################################
    #############################################################################################
    ## Misc FUNCTIONS 
    ##

    def compute_vigilance(self,vigilance_value):
        vigilance_matrix = self.i_ext*0
        vigilance_matrix[:,10:] = vigilance_value
        return vigilance_matrix



