

import numpy as np
import matplotlib.pyplot as plt



def build_histo(data,x_step:int=1,nbins:int=100,offset:int=0,data_range_factor:int=.1,**kwargs):
    
    ###############################################################
    ## Build heatmap from data. 
    ## params xstep is the legnth step, must be an integer
    ## params nbins is the number of y bins
    ## offset is used for the xticks, to define the offset time
    ## smoothing window smooth the data using a window kernel
    sh = data.shape
    reshaped_data = data.T.reshape((sh[0]*sh[1],))
    
    ###############################################################
    # ymax and ymin, if not given are computed from the data
    ymax = np.nanmax(data)
    ymin = np.nanmin(data)
    data_range = ymax - ymin
    
    if 'ymax' in kwargs.keys():
        ymax = kwargs['ymax']
    else:
        ymax += data_range*data_range_factor
    if  'ymin' in kwargs.keys():
        ymin = kwargs['ymin']
    else:
        ymin -= data_range*data_range_factor
        
    ###############################################################
    #tiles for the heatmap
    tiles = np.tile(np.arange(0,sh[0],1),sh[-1])

    ###############################################################
    #bins for the heatmap
    xbins = np.arange(0,sh[0],x_step)
    ybins = np.linspace(ymin,ymax,nbins+1)
    
    ###############################################################
    #ticks for the graph
    xticks = np.arange(0, sh[0]-x_step,x_step)-offset
    yticks = np.linspace(ymin,ymax,nbins)
    yticks = np.around(yticks,3)
    
    heatmap, xbins, ybins = np.histogram2d(tiles,reshaped_data,bins=(xbins,ybins),density=True)
    return heatmap[1:-2,:],xticks[1:-2],yticks,ybins,xbins[1:-2],ymin,ymax

def add_hline(y,ax,xticks,color='red'):
    ones = np.ones((len(xticks),))
    ones *= y
    ax.plot(xticks,ones,color=color)
    
def add_vline(x,ax,yticks,color='red'):
    ones = np.ones((len(yticks),))
    ones *= x
    ax.plot(ones,yticks,color=color)

def master_plot(data,show_xaxis:bool=True,show_yaxis:bool=True,grid_style:str=':',levels:int=10,
                plot_average:int=True,print_sessions:int=0,offset:int=0,smoothing_window:int=1,
                title=None,additional_plots=[],**kwargs):
     
    ###############################################################
    #Smoothing data
    #data = sde.array_smoother(data,smoothing_window)
    
    ###############################################################
    # Initiating graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.grid(linestyle=grid_style)
    if title is not None:
        ax.set_title(title)
    
    ###############################################################
    #Computing Heatmap
    heatmap,xticks,yticks,ybins,xbins,ymin,ymax=build_histo(data,offset=offset,**kwargs)

    ###############################################################
    # Plotting Heatmap
    ax.contourf(xticks,yticks,heatmap.T, levels=levels, cmap='coolwarm')
    
    ###############################################################
    # Plotting the origin lines  
    if show_xaxis:
        add_hline(0,ax,xticks,color='black')
    if show_yaxis:
        add_vline(0,ax,yticks,color='black')
        
    ###############################################################
    # Plotting the average of the data
    full_xticks = np.arange(xticks[0], xticks[-1]+1,1)
    if plot_average:
        md = np.clip(np.nanmean(data,1),ymin,ymax)
        ax.plot(full_xticks,md[full_xticks[0]+offset:full_xticks[-1]+offset+1],
                color='red',label='Average over sessions')
            
    ###############################################################
    # Plotting the average of the data
    colors = ('darkred','darkblue','black')
    data = np.clip(data,ymin,ymax)
    for i in range(np.clip(print_sessions,0,3)):
        ax.plot(full_xticks,data.T[i,full_xticks[0]+offset:full_xticks[-1]+offset+1],color=colors[i],
                linewidth=.2,label='Sample session')
        
        
    ###############################################################
    # Plotting additional data
    count = 0
    colors = ['blue','purple','orange','green'] 
    for data,title in additional_plots:
        ax.plot(full_xticks,data[full_xticks[0]+offset:full_xticks[-1]+offset+1],color=colors[count%len(colors)],label=title)
        count +=1


    ax.legend()
    plt.show()
    return heatmap,xticks,xbins,yticks,ybins,md,data,full_xticks
