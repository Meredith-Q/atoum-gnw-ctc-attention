###################################################################
###################################################################
## 
## Misc Functions related to the Model and MultipleSimulations
## 




###################################################################
###################################################################
## IMPORT SECTION 
## 
## 

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/../toolbox')
notebook_relative_location = "../"


import numpy as np

from brian2 import second, ms, pA, Hz



sys.path.insert(0,parentdir+'/../genetic/genetic_v2/')

from multiple_simulations import MultipleSimulations

from model_class import Model


from tqdm.notebook import tqdm


import plotly as plty
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

import hashlib
import pickle

###################################################################
## Building Functions to be recorded by the MultipleSimulations object
## based on network parameters

def get_functions_from_params(network_params):
    area_number = network_params['area_number']
    data_shape = (int(network_params["run_t"]/network_params["dt"])+1,area_number)
    return {
                'firing_rates':{
                'function': (lambda m : m.get_r()),
                'args' : {},
                'type':"at_the_end",
                'data_shape': data_shape,
            },
                'firing_rates2':{
                'function': (lambda m : m.get_r(2)),
                'args' : {},
                'type':"at_the_end",
                'data_shape': data_shape,
            },
    }


###################################################################
## Launching simulations for a given set of paraetemers 

def launch_simu(network_params,simulation_number=1,talky=False,**kwargs):
    network_params = dict(network_params)
    network_params.update(kwargs)
    functions = get_functions_from_params(network_params)
    kwargs = {'network_params':dict(network_params),'talky':False}
    recording_step = 1
    simulation_steps= 1
    simulation_number = simulation_number

    mss = MultipleSimulations(Model,functions,object_kwargs=kwargs,recording_step=recording_step,
                             simulation_steps=simulation_steps,simulation_number=simulation_number,talky=talky,
                             )
    mss.run()
    return mss

    
###################################################################
##  
## Hashing functions 
def remove_unit(value):
    try:
        return value/value.get_best_unit()
    except AttributeError:
        return value

def bin_list_to_str(list_value):
    result = ""
    for value in list_value:
        result += value.decode()
    return result

def value_to_string(value):
    if type(value) == list:
        return bin_list_to_str(value)
    else:
        temp = remove_unit(value)
        if type(temp) == str:
            return temp
        else:
            return str(np.round(temp,5))

def kwargs_to_string(kwargs,sep=";;;"):        
    hash_string = ""
    sorted_keys = list(kwargs.keys())
    sorted_keys.sort()
    for key in sorted_keys:
        hash_string += sep + key + value_to_string(kwargs[key])
    return hash_string

def hash_text(text,is_bytes=False):
    # Generic hashing function

    if not is_bytes:
        text = text.encode()
    return hashlib.sha256(text).hexdigest()

def hash_file(filename):
    # File hashing function

    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    try:
        with open(filename, 'rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                h.update(mv[:n])
    except FileNotFoundError:
        with open(notebook_relative_location+filename, 'rb', buffering=0) as f:
            for n in iter(lambda : f.readinto(mv), 0):
                h.update(mv[:n])

    return h.hexdigest()

def hash_kwargs(kwargs):
    # Dict hashing function

    kwargs_hash = kwargs_to_string(kwargs)

    return hash_text(kwargs_hash,False)

def compute_hash_from_kwargs(kwargs,file_name="model_class.py"):
    # Kwargs hashing function, assumes file is model_class.py

    model_hash = hash_file(file_name)
    template_kwargs_hash = hash_kwargs(kwargs)

    return hash_text(model_hash+template_kwargs_hash)

###################################################################
###################################################################
## 
## Updating Network Params with Variable Steady States 
## 

###################################################################
## Updating Network Params with variable steady states as initial
## values

def build_full_kwargs(network_params,connectivity_file):
    area_number = network_params['area_number']
    ## Build kwargs finding the empirical steady states
    limits = {
        "firing_rates":np.zeros((area_number,)),
        "firing_ratesi":np.zeros((area_number,)),
        "s_i":np.zeros((area_number,)),
        "s_1a":np.zeros((area_number,)),
        "s_1n":np.zeros((area_number,)),
    }
    network_params_from_limits(limits,connectivity_file,network_params)
    limits = find_limits(area_number,network_params)
    network_params_from_limits(limits,connectivity_file,network_params)

def get_full_kwargs(kwargs):

    temp_kwargs = dict(kwargs)

    dynamic_model_keys = [
        'attention_baseline_gain', 'attention_task_gain',
        'attention_task_tau', 'attention_task_on', 'attention_task_off',
        'attention_reentry_gain',
        'attention_ignition_threshold_hz', 'attention_ignition_slope_hz',
        'attention_reentry_current', 'attention_sensory_floor_hz',
        'workspace_areas',
    ]
    temp_keys = ['sigma_e','sigma_i','run_t','stim_on','stim_off']
    temp_keys.extend(key for key in dynamic_model_keys if key in kwargs)
    for key in temp_keys:
        temp_kwargs.pop(key)

    to_remove_keys = ['r_init', 'ri_init', 's_i_init', 's_1a_init', 's_2a_init', 's_1n_init', 's_2n_init']
    for key in to_remove_keys:
        try:
            del temp_kwargs[key]
        except KeyError:
            pass

    full_hash = compute_hash_from_kwargs(temp_kwargs)

    if os.path.exists('data/full_kwargs'):
        try:
            with open("data/full_kwargs/"+full_hash+'.pickle', 'rb') as handle:
                full_kwargs = pickle.load(handle)

        except FileNotFoundError:
            full_kwargs = dict(kwargs)
            build_full_kwargs(full_kwargs,full_kwargs["connectivity_file"])
            with open("data/full_kwargs/"+full_hash+'.pickle', 'wb') as handle:
                pickle.dump(full_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(notebook_relative_location+"data/full_kwargs/"+full_hash+'.pickle', 'rb') as handle:
                full_kwargs = pickle.load(handle)

        except FileNotFoundError:
            full_kwargs = dict(kwargs)
            build_full_kwargs(full_kwargs,full_kwargs["connectivity_file"])
            with open(notebook_relative_location+"data/full_kwargs/"+full_hash+'.pickle', 'wb') as handle:
                pickle.dump(full_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for key in temp_keys:
        full_kwargs[key] = kwargs[key]
    return full_kwargs

    

###################################################################
## Finding variable steady states 

def find_limits(area_number,network_params):
    ## Let the network at rest for a few seconds to find the steady states of multiple variables

    run_t = 5 * second

    functions = {
                    'firing_rates':{
                    'function': (lambda m : m.get_r()),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    'firing_rates2':{
                    'function': (lambda m : m.get_r(2)),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    'firing_ratesi':{
                    'function': (lambda m : m.get_ri()),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    's_i':{
                    'function': (lambda m : m.get_si()),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    's_1n':{
                    'function': (lambda m : m.get_sn()),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    's_2n':{
                    'function': (lambda m : m.get_sn(2)),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    's_1a':{
                    'function': (lambda m : m.get_sa()),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
                    's_2a':{
                    'function': (lambda m : m.get_sa(2)),
                    'args' : {},
                    'type':"at_the_end",
                    'data_shape': (int(run_t/network_params["dt"])+1,area_number),
                },
            }


    kwargs = {'network_params':dict(network_params)}
    recording_step = 1
    simulation_steps= 1
    simulation_number = 1
    kwargs["network_params"]["i_ext"] = 0 *pA
    kwargs["network_params"]["run_t"] = run_t
    kwargs["network_params"]["sigma_e"] = 0*pA
    kwargs["network_params"]["sigma_i"] = 0*pA
    kwargs['talky'] =False

    mus = MultipleSimulations(Model,functions,object_kwargs=kwargs,recording_step=recording_step,
                             simulation_steps=simulation_steps,simulation_number=simulation_number,talky=False)

    mus.run()
    result = {}
    for key in list(functions.keys()):
        result[key] = mus.data[key][0,-1]
    return result


###################################################################
## Updating network params with the limits (they have to be computed
## already)

def network_params_from_limits(limits,connectivity_file,network_params):
    ## update network params with the computed steady states
    network_params["r_init"] = limits["firing_rates"]*Hz
    network_params["ri_init"] = limits["firing_ratesi"]*Hz
    network_params["s_i_init"] = limits["s_i"]
    network_params["s_1a_init"] = limits["s_1a"]
    network_params["s_2a_init"] = limits["s_1a"]
    network_params["s_1n_init"] = limits["s_1n"]
    network_params["s_2n_init"] = limits["s_1n"]
    network_params["connectivity_file"] = connectivity_file


###################################################################
###################################################################
## 
## Latency functions
## 

###################################################################
## Finding Latencies for a given data set (Defined as time to reach
## 90% of the maximum activity)

def get_latency(data,index=1):
    temp = data.T > np.clip(np.max(data,(index,)).T*.9,10,500)
    temp = np.argmax(temp.T,1)
    temp = temp.astype('float')
    temp[temp == 0] = np.nan
    return temp

###################################################################
## Finding Latencies for a given MultipleSimulations object
##(Defined as time to reach 90% of the maximum activity)

def get_latency_from_ms(mss):
    data = mss.data['firing_rates'][0]
    temp = data.T > np.clip(np.max(data,(1,)).T*.9,10,500)
    temp = np.argmax(temp.T)
    return temp


###################################################################
###################################################################
## 
## Pyschometric curve function 
## 

###################################################################
##  
## Finding, for a given Network Params the input current range
## for wich the network shows bitable activity
##  

def get_y_from_network_params(network_params,y_min=0*pA,y_max=400*pA,eps=.1*pA,talky=False):
    y_int = None
    
    d = y_max-y_min
    
    
    result = dict()
    latency = dict()
    
    pbar = tqdm(total=int(2*np.log2((y_max-y_min)/eps)+2))

    ############################################################################
    ## Checking if resting state is stable
    #a = sustained_activity(y_min,network_params)
    kwargs = {'i_ext':y_min}
    mss = launch_simu(network_params,simulation_number=10,**kwargs)
    a = sustained_activity(mss)

    pbar.update()

    if a==1.:
        print('Unstable resting state')
        return y_min,y_min,result
    
    result[str(y_min/pA)] = a

    ############################################################################
    ## Checking if excited state exists
    #b = sustained_activity(y_max,network_params)
    kwargs = {'i_ext':y_max}
    mss = launch_simu(network_params,simulation_number=1,**kwargs)
    b = sustained_activity(mss)

    pbar.update()
    
    if b==0.:
        print('No excited state')
        return y_max,y_max,result

    result[str(y_max/pA)] = b
    latency[str(y_max/pA)] = get_latency_from_ms(mss)

    while d > eps:
        y_new = (y_min+y_max)/2
        #f = sustained_activity(y_new,network_params,simulation_number=10)

        kwargs = {'i_ext':y_new}
        mss = launch_simu(network_params,simulation_number=10,**kwargs)
        f = sustained_activity(mss)

        pbar.update()
        result[str(y_new/pA)] = f

        if f > 0.:
            latency[str(y_new/pA)] = get_latency_from_ms(mss)

        
        ##########################################################################
        ## Checking if point found in the middle of the sigmoid
        if f > 0. and f <1.:
            y_int = y_new
            if talky:
                print('Mashala! ' + str(y_new))

            ######################################################################
            ## Looking for y_max2
            y_min1 = y_int
            y_max1 = y_max
            d1 = y_max1-y_min1
            while d1 > eps:
                y_new = (y_min1+y_max1)/2

                kwargs = {'i_ext':y_new}
                mss = launch_simu(network_params,simulation_number=10,**kwargs)
                f = sustained_activity(mss)

                #f = sustained_activity(y_new,network_params,simulation_number=10)
                pbar.update()
                result[str(y_new/pA)] = f
                if f > 0.:
                    latency[str(y_new/pA)] = get_latency_from_ms(mss)

                if f == 1.:
                    y_max1 = y_new
                else :
                    y_min1 = y_new
                d1 = y_max1-y_min1
                if talky:
                    print(str(y_new) + ' : ' + str(f))
            y_max2 = y_new

            #####################################################################
            ## Looking for y_min2
            y_min1 = y_min
            y_max1 = y_int
            d1 = y_max1-y_min1
            while d1 > eps:
                y_new = (y_min1+y_max1)/2

                kwargs = {'i_ext':y_new}
                mss = launch_simu(network_params,simulation_number=10,**kwargs)
                f = sustained_activity(mss)

                #f = sustained_activity(y_new,network_params,simulation_number=10)
                pbar.update()
                result[str(y_new/pA)] = f
                if f > 0.:
                    latency[str(y_new/pA)] = get_latency_from_ms(mss)

                if f == 0.:
                    y_min1 = y_new
                else :
                    y_max1 = y_new
                d1 = y_max1-y_min1
                if talky:
                    print(str(y_new) + ' : ' + str(f))
            y_min2 = y_new
            
            return y_min2,y_max2,result,latency
        elif f == 1.:
            y_max = y_new
        elif f == 0.:
            y_min = y_new
            
        d = y_max-y_min
        if talky:
            print(str(y_new) + ' : ' + str(f))
    return y_min,y_max,result,latency


###################################################################
##  
## Plotting Psychometric curve
##  

def plot_sig(b,d,result={},title=None,x_min=0,x_max=50):

    def sig(x,b,d):
        return 1/(1+np.exp(-d*(x-b)))

    xx = np.array(list(result.keys()))
    yy = np.array(list(result.values()))
    
    xx = xx.astype(float)

    fig = go.Figure(data=go.Scatter(x=xx, y=yy, mode='markers'))



    fig.update_layout(title=title,
                      autosize=False,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90),
                        xaxis = dict(
                            range=[x_min,x_max],
                            title='Input current (pA)',),
                        yaxis = dict(
                            range=[0,1.5],
                            title='Detection'),
                     )

    xx1 = np.linspace(x_min,x_max,200)
    fig.add_trace(go.Scatter(x=xx1,y=sig(xx1,b,d), mode='lines'))
    fig.show()
     

###################################################################
##  
## Computing Psychometric curve bounds for a given set of parameters
##  

def get_sig_from_params(network_params,x_min=0*pA,x_max=400*pA,eps=.1*pA,show=False,title=None,
        talky=False):
    y_min,y_max,result,latency = get_y_from_network_params(network_params,x_min,x_max,eps,talky)

    b = (y_min+(y_max-y_min)/2)
    d = 1/(y_max-y_min)*4


    b,d = b/pA,d*pA
    if show:
        plot_sig(b,d,result,title=title,x_min=x_min/pA,x_max=x_max/pA)
   
    return b,d,latency


###################################################################
##  
## testing if the area of a MultipleSimulations is showing 
##  Persistent activity


def sustained_activity(mss,area=0):
    a = mss.data['firing_rates'][:,-1,area]
    return np.mean(a > 10)



###################################################################
##  
##  Not sure what's happening there 


def plot_bf(ss,INN,w,col=None,row=None,fig=None):

    if fig is None:
        fig = go.Figure()
        show=True
    else:
        show = False


    if show:
        layout = go.Layout(
                    #xaxis=dict(range=[0,.16]),
                    #yaxis=dict(range=[0,.16]),
                    xaxis_title="I_A",
                    yaxis_title="I_B",
                    title=r'I_1^A\ I_1^B Bifurcation Diagrams',
                    height=600,
                    width=600,
                )

        fig.update_layout(layout)

    fig.add_trace(
            go.Scatter(x=INN, y=ss*w,name="Area A bifurcation curve",line=dict(color="blue")),col=col,row=row)
    fig.add_trace(
            go.Scatter(x=ss*w, y=INN,name="Area B bifurcation curve",line=dict(color="orange")),col=col,row=row)

    if show:
        fig.show()


###################################################################
##  
##  Not sure what's happening there 


def plot_bf_from_params(network_params,w=.2,i_max=50,m=20):
    
    mp1 = {
        'i_ext' : (0,i_max,m,pA),
    }
    
    custom_kwargs = dict(network_params)
    
    custom_kwargs['sigma_e'] = 0 * pA 
    custom_kwargs['sigma_i'] = 0 * pA 

    build_full_kwargs(custom_kwargs,'subgraph_data30.pkl')
    functions = get_functions_from_params(custom_kwargs)
    kwargs = {'network_params':dict(custom_kwargs),'talky':False}
    recording_step = 0
    simulation_steps= 1
    simulation_number = 1

    mss1 = MultipleSimulations(Model,functions,object_kwargs=kwargs,recording_step=recording_step,
                             simulation_steps=simulation_steps,simulation_number=simulation_number,
                             moving_params=mp1,talky=True)

    mss1.run()
    fr = mss1.data['firing_rates'][:,0,:,0]
    fr_max = np.max(fr,1)
    
    INN = np.linspace(0,i_max,20)
    
    plot_bf(fr_max,INN,w)
    
    return fr_max
    
