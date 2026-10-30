


from timeit import time
import numpy as np



class Timer():



    def __init__(self,keys,n):

        self.__dic__ = {}
        self.init = {}

        for key in keys:
            self.__dic__[key] = np.zeros((n,))
            self.init[key] = 0

    def initiate(self,key):
        self.init[key] = time.time()

    def add_time(self,key,it):
        self.__dic__[key][it] = time.time() - self.init[key]

    def get_time(self,key):
        return self.__dic__[key]

    def get_avg_time(self,key):
        return np.mean(self.__dic__[key])

    def get_avg_times(self):

        result_dic = {}
        for key in self.__dic__.keys():
            result_dic[key] = np.mean(self.__dic__[key])
        return result_dic
