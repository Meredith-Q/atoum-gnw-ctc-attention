


import numpy as np

from brian2 import second, ms, pA, Hz




template_kwargs = {
    ###############################################################
    ## Time constants and synapses parameters

    'tau_r': 2 * ms,  # firing rate decay time
    'tau_a': 2 * ms,  # AMPA decay time
    'tau_g': 5 * ms,  # GABA decay time
    'tau_n': 60 * ms, # NMDA decay time
    'gamma_n': 2 * 0.641, # NMDA gamma
    'gamma_a': 2,         # AMPA gamma
    'gamma_g': 2,         # GABA gamma
    'c_a' : 1 * pA,# AMPA conductance (Unit free, ##FIXME)
    'c_n' : 1 * pA, # NMDA Conductance

    ###############################################################
    ## threshold linear firing rate transfer function parameters

        ## Excitatory parameters
        'a_e': .5 * 0.27 * Hz / pA,
        'b_e': .5 * 108 * Hz,
        'd_e': 2 * 0.154 * second,

        ## Inhibitory parameters
        'g_i': 4.0,
        'c_1': 0.615 * Hz / pA,
        'c_0': 177 * Hz,
        'r_0': 5.5 * Hz,

    ###############################################################
    ## Background input and noise

    'i_bg_e': 329.4 * pA, # Constant Input to Excitatory units
    'i_bg_i': 260 * pA,   # Constant Input to Inhibitory units
    'sigma_e': 0 * pA,  # Noise Variance in Excitatory units
    'sigma_i': 0 * pA,    # Noise Variance in Inhibitory units

    ###############################################################
    ## External input

    'active_area' : [b'V1'], # Target Area
    'i_ext': 200 * pA,      # External current
    'stim_on': .01 * second, # Stimulus onset
    'stim_off': .06* second, # Stimulus offset

    ###############################################################
    ## Simulation parameters

    'run_t': 1 * second, # Simulation running time
    'dt': .5 * ms,       # Simulation time step

    "fixed_vars": [], # fixed variables, won't be updated

    ###############################################################
    ## Network parameters

    "area_number" : 40,
    'gN': 1000*pA,                   # Network longrange strength
    'gA': 10000*pA,                  # Network longrange strength

    ###############################################################
    ## Optional attentional gain and sensory re-entry (disabled by default)

    'attention_baseline_gain': 1.0,
    'attention_task_gain': 0.0,
    'attention_task_tau': 50 * ms,
    'attention_reentry_gain': 0.0,
    'attention_ignition_threshold_hz': 20 * Hz,
    'attention_ignition_slope_hz': 2 * Hz,
    'attention_reentry_current': 0 * pA,
    'attention_sensory_floor_hz': 5 * Hz,

    'fln_a': 1.2, # Fln factor
    'fln_b': 0.3, # Fln factor (fln = c_a*fln**c_b)

    "connectivity_file" : 'subgraph_data40.pkl',
    "spine_count_file" : "spine_count_40_areas.npy",

    ###############################################################
    ## Local connections strength

    'G':1.5,
    'Gi':.105,

    'kappa_loc':.90909,
    'g_s': .484,
    'g_c': 0,

    'z_min':.6,
    'zI_min':.218,

    'g_0': 215*pA,

    'g_ie' : .01,
    
    'j_gei': -8800 * pA, # Local Inhibitory to Excitatory connection strength
    'j_gii': -120 * pA,  # Local Inhibitory to Inhibitory connection strength


    ###############################################################
    ## Long-range connections strength

    #'lr_sp_min' : .77, # Minimum spine count for long range connections
    #'lr_sp_max' : 1, # Maximum spine count for long range connections

    ###############################################################
    ## Nus

    'nu_ff_e' : 1,
    'nu_ff_n' : 0,
    'nu_fb_e' : .015,
    'nu_fb_n' : .8,

}

