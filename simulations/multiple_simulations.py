

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from scipy import stats



from simulation import Simulation


from super_plot import * 

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/nyu/toolbox')


from density_plot import *


class MultipleSimulations:


    ################################################################################################################
    ################################################################################################################
    ##
    ## Init function
    ##


    def __init__(self,object_class,functions_to_monitor,object_kwargs={},
            recording_step=1,simulation_steps=1,simulation_number=1,talky=True,
            moving_params={},notebook=False):
        ##############################################################################
        ##
        ## moving_params = { 'key1':(min1,max1,step_nb1,unit1),
        ##                      'key2':(min2,max2,step_nb2,unit2),
        ##      }
        self.object_class = object_class
        self.object_kwargs = object_kwargs
        self.functions_to_monitor = functions_to_monitor
        self.recording_step = recording_step
        self.simulation_steps = simulation_steps
        self.simulation_number = simulation_number
        self.simulations = []
        self.talky = talky
        self.moving_params = moving_params
        self.notebook = notebook

        self.data = {}

        steps = list(map(lambda x: moving_params[x][2],moving_params.keys())) 



    

        for function in self.functions_to_monitor.keys():
            self.data[function] = []
            f = self.functions_to_monitor[function]
            if f["type"] == "on_the_go":
                self.data[function] = np.zeros((*steps,simulation_number,
                    int(simulation_steps/recording_step)))
            elif f["type"] == "at_the_end":
                data_shape = self.functions_to_monitor[function]["data_shape"]
                self.data[function] = np.zeros((*steps,simulation_number,*data_shape))


    ################################################################################################################
    ################################################################################################################
    ##
    ## Running functions 
    ##


    def run(self):
        if self.talky:
            total_iterations = self.simulation_number
            for key in self.moving_params.keys():
                total_iterations *= self.moving_params[key][2]

            if self.notebook:
                self.pbar = tqdm(total=total_iterations, desc="Model simulations", unit="run")
            else:
                self.pbar = tqdm_notebook(
                    total=total_iterations, desc="Model simulations", unit="run"
                )
        self.run_loop(self.moving_params,[],[],[])


    def core_run(self,indexes,values,var_names):
        custom_range = range(self.simulation_number)

        for i in custom_range:
            object_kwargs = dict(self.object_kwargs)
            object_kwargs['network_params'] = dict(object_kwargs['network_params'])
            for index,key in enumerate(var_names):
                object_kwargs["network_params"][key] = values[index]
            object = self.object_class(**object_kwargs)
            s = Simulation(object,self.functions_to_monitor,
                    recording_step=self.recording_step,talky=False)
            s.run(self.simulation_steps)
            self.simulations.append(s)

            for function in self.functions_to_monitor.keys():
                f = self.functions_to_monitor[function]
                if f["type"] == "on_the_go":
                    slices = [*indexes,i,(len(self.data[function].shape)-len(indexes)-1)] 
                    self.data[function][slices] = s.data[function] 
                elif f["type"] == "at_the_end":
                    slices = tuple([*indexes, i, *[slice(None)] * (len(self.data[function].shape) - len(indexes) - 1)]) 
                    self.data[function][tuple(slices)] = s.data[function]
            if self.talky:
                self.pbar.update()

    def run_loop(self,moving_params,indexes,values,var_names):
        if len(moving_params) == 0:
            self.core_run(indexes,values,var_names)
        else:
            key = list(moving_params.keys())[0]
            params = moving_params[key]
            param_range = np.linspace(params[0],params[1],params[2])*params[3]
            moving_params_bis = dict(moving_params)
            moving_params_bis.pop(key)
            for i,j in enumerate(param_range):
                self.run_loop(moving_params_bis,(*indexes,i),(*values,j),(*var_names,key))

    ################################################################################################################
    ################################################################################################################
    ##
    ## Getting functions 
    ##

    def get_data(self,function_name):
        sample = self.simulations[0].data[function_name][0]
        simu_number = len(self.simulations)
        r = np.zeros((*sample.shape,simu_number))

        for i,simu in enumerate(self.simulations):
            r[:,:,i] = self.simulations[i].data[function_name][0]
        return r


    def plot_density(self,**kwargs):
        
        master_plot(self.data.T,**kwargs)

    def get_steady_state(self,data,bin_number,window_size,dt=1,p_value=.0001,show=False):
    
        bins = int(data.shape[1]/bin_number)
        c = np.zeros(bins)

        for i in range(bins):
            b = np.mean(data[:,bin_number*i:bin_number*i+window_size],1)
            b1 = np.mean(data[:,bin_number*i+window_size:],1)
            c[i] = stats.ttest_ind(b,b1).pvalue
        
        d=np.minimum(c,p_value)
        if show:
            plt.plot(d)
            plt.show()
        valid_idx = np.where(c < p_value)[0]
        return (np.max(valid_idx) +1)*bin_number*dt

    def get_mean_value_on_moving_param(self,function_name,onset=0,offset=-1,moving_param_slice=None):
        if moving_param_slice is None:
            moving_param_slice = slice(None)
        data = self.data[function_name]
        shape = data.shape
        ## Selecting the portion between onset and offset
        slices = (moving_param_slice,slice(None),slice(onset,offset),*[slice(None)]*(len(shape)-3))
        d = data[slices]
        ## Averaging on time and simulations
        return np.mean(d,(1,2))

    def get_max_value_on_moving_param(self,function_name,onset=0,offset=-1,moving_param_slice=None):
        if moving_param_slice is None:
            moving_param_slice = slice(None)
        data = self.data[function_name]
        shape = data.shape
        ## Selecting the portion between onset and offset
        slices = (moving_param_slice,slice(None),slice(onset,offset),*[slice(None)]*(len(shape)-3))
        d = data[slices]
        ## Averaging on time and simulations
        return np.max(d,(1,2))

    ################################################################################################################
    ################################################################################################################
    ##
    ## Computing functions 
    ##

    def matching_areas(self,testing_functions,details=False):
        #################################################
        ## Function designed for space parameter exploration
        ## computing_functions must be a list of function, turning simulations into real value
        ## testing_functions must be a list of function, turning real value to boolean value
        ## Return logical and of all testing functions applied on computing functions

        n = len(testing_functions)
        boules = []
        for i in range(n):
            #data = computing_functions[i](self)
            boules.append(testing_functions[i](self))
        if details:
            return boules
        else:
            return np.all(boules,axis=0)

    def max_value_moving_params(self,data):
        key_number = len(self.moving_params.keys()) # Number of moving params
        data_dim = len(self.data[list(self.data.keys())[0]].shape) # Number of dimensions of data
        return np.nanmax(data,tuple(list(range(key_number,data_dim)))) # mean for every moving param

    def steady_state_value_moving_params(self,data,time_key_position=-1):
        raise Error('BROKEN FUNCTION, TO BE FIXED')
        """
        key_number = len(self.moving_params.keys()) # Number of moving params
        data_dim = len(self.data[list(self.data.keys())[0]].shape) # Number of dimensions of data
        if time_key_position < 0:
            time_key_position = data_dim + time_key_position# If time_key is -1, it should be set to the data_dim for the next formula to work
        steady_states = data[(*[slice(None)]*(time_key_position-2)),-1,(*[slice(None)]*(data_dim-time_key_position))] # extracting the last temporal value of each trial
        return np.nanmean(data,tuple(list(range(key_number,data_dim)))) # Returning the mean
        """



    ################################################################################################################
    ################################################################################################################
    ##
    ## Plotting functions 
    ##

    def plot_moving_params(self,data_key,computing_function,fig_size=(15,5),params={},display=True,ticks_round=None,
            title=None):
        #################################################
        ## Plot function take a multiple session with up to four different moving params
        ## will plot the heat map in four dimensions
        ## data_key will help gather the considered data (ms.data[data_key])
        ## plotting_function will take a 2D numpy array and plot a heatmap
        ## computing_function transforms the data to compute any desired value
        ## params are custom params like supplementary dimensions

        keys = list(self.moving_params.keys())
        key_number = len(keys)
        ticks = list(map(lambda x :
                         np.round(np.linspace(self.moving_params[x][0],self.moving_params[x][1],self.moving_params[x][2]),1),
                        keys))

        if ticks_round is not None:
            for i in range(key_number):
                ticks[i] = np.round(ticks[i],ticks_round)

        fig = plt.figure(constrained_layout=True,figsize=fig_size)
        fig.suptitle(title)

        data = self.data[data_key]
        computed_data = computing_function(data)

        if display:
            if key_number == 1:
                plt.plot(computed_data)

            if key_number == 2:
                self.plot_heatmap(computed_data[:,:],
                                 xlabel=keys[0],ylabel=keys[1],xticks=ticks[0],yticks=ticks[1])

            if key_number == 3:
                gs = fig.add_gridspec(1,self.moving_params[keys[0]][2])
                for i in range(self.moving_params[keys[1]][2]):
                    ax = fig.add_subplot(gs[i])
                    self.plot_heatmap(computed_data[i,:,:],
                                 ax=ax,xlabel=keys[1],ylabel=keys[2],xticks=ticks[1],yticks=ticks[2])
                    ax.set_title(keys[0] + " : " + str(ticks[0][i]))

            if key_number == 4:
                gs = fig.add_gridspec(self.moving_params[keys[0]][2], self.moving_params[keys[1]][2])
                for i in range(self.moving_params[keys[0]][2]):
                    for j in range(self.moving_params[keys[1]][2]):
                        ax = fig.add_subplot(gs[i, j])
                        self.plot_heatmap(computed_data[i,j,:,:],
                                     ax=ax,xlabel=keys[2],ylabel=keys[3],xticks=ticks[2],yticks=ticks[3])
                        ax.set_title(keys[0] + " : " + str(ticks[0][i]) + ", " + keys[1] + " : " + str(ticks[1][j]))


        return computed_data

    def plot_heatmap(self,data,title="",ax=None,xticks=None,yticks=None,xlabel="",ylabel=""):
        ax = sns.heatmap(data.T,ax=ax)
        shape = data.shape
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is None:
            ax.set_xticklabels(np.linspace(0,1,shape[0]))
        else:
            ax.set_xticklabels(xticks)
        if yticks is None:
            ax.set_yticklabels(np.linspace(0,1,shape[1]))
        else:
            ax.set_yticklabels(yticks)
