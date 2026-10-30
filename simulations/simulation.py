

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm_notebook as tqdm


from super_plot import *



class Simulation:


    def __init__(self,object_to_run,functions_to_monitor,recording_step=1,plotting_step=1,talky=True):
        ## object_to_run must have a run() function that will be computed here
        ## functions_to_monitor must be a dict filled with functions
        ## functions_to_monitor = {
        ##  "foo1":{"function":lambda x : x,"kwargs":{}},
        ##  "foo2":{"function":lambda x : 2*x,"kwargs":{}},
        ## }
        self.object = object_to_run
        self.object.simulation  = self
        self.functions = functions_to_monitor
        self.data = {}
        self.plotting_on_the_go_functions = []
        for function in self.functions.keys():
            self.data[function] = []
            f = self.functions[function]
            if f["type"] == "plotting_on_the_go":
                self.plotting_on_the_go_functions.append(f["function"])
        self.recording_step = recording_step
        self.plotting_step = plotting_step
        self.talky=talky



    def run(self,simulation_steps=1):
        if self.talky:
            r = tqdm(range(simulation_steps))
        else:
            r = range(simulation_steps)
        for i in r:
            self.simulation_step = i
            exception = self.object.run()
            if exception == -1:
                print("Simulation step : " + str(i))
                return -1
            if i % self.recording_step == 0:
                for function in self.functions.keys():
                    if self.functions[function]["type"] == "on_the_go":
                        args = self.functions[function]["args"]
                        foo = self.functions[function]["function"]
                        result = foo(self.object,*args) 
                        self.data[function].append(result)
            if i % self.plotting_step == 0:
                for function in self.plotting_on_the_go_functions:
                  function()
        for function in self.functions.keys():
            if self.functions[function]['type'] == 'at_the_end':
                args = self.functions[function]["args"]
                foo = self.functions[function]["function"]
                result = foo(self.object,*args) 
                self.data[function] = result


    def simple_plot(self,function_name,timing_slice=None):
        if timing_slice is None:
            timing_slice = slice(None)
        plt.plot(self.data[function_name][timing_slice])
        plt.title(function_name)
        plt.show()      

    def distrib_plot(self,function_name):
        df = pd.DataFrame()

        for i in range(len(self.data[function_name])):
            data = self.data[function_name][i]
            df2 = pd.DataFrame({"food":data,'step':[i]*len(data)})
            df = df.append(df2)
    
        sns.scatterplot(x="step",y="food",data=df)
        plt.title(function_name)

    def distrib_plot2(self,function_name,smoothing=10,mmin=1,mmax=0,a_size=100):
        foo(self.data[function_name],smoothing,mmin,mmax,a_size)        
        plt.title(function_name)

    def get_fixation_time(self,gene_name):
        data = np.array(self.data[gene_name+"_proportion"])
        inf = np.argmin(data>.001)
        if inf == 0 and data[0] > .001:
            inf = None 
        sup = np.argmin(data<.999)
        if sup == 0 and data[0] < .999:
            sup = None
        if inf is None and sup is None:
            return None
        elif inf is None:
            return sup*self.recording_step
        elif sup is None:
            return inf*self.recording_step
        else:
            return min(sup,inf)*self.recording_step
