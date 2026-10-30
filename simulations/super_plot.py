


import numpy as np


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)



def foo(a,smoothing,mmax,mmin,array_size):
	step = (mmax-mmin)/array_size
	vect = np.arange(mmin,mmax+step,step)[:array_size+1]

	d = get_smoothed_hist(a,smoothing,array_size,mmin,mmax)
	fig = _core_figure_2(d,"l","MUA",vect,timing_window=[0,None],three_d=False)
	iplot(fig, filename='basic-line')
	return d



def array_smoother(data,window=50):
	#Function smoothing all the vector of a 2d numpy array
	smoothed_data = np.apply_along_axis(lambda x : smoother(x,window),0,data)
	return smoothed_data



def smoother(data,window=10):
	#TO BE FIXED, nice way to smooth still have edge effects
	box = np.ones(window)/window
	result = np.convolve(data,box,mode='same') 
	return result


		
def get_smoothed_hist(data,smoothing,array_size,mmin,mmax):

	step = (mmax-mmin)/array_size
	vect = np.arange(mmin,mmax+step,step)[:array_size+1]

	#Building histogram
	step_number = len(data)
	c = np.zeros((step_number,array_size))
	for i,x in enumerate(data):
		a,b = np.histogram(x,vect,density=True)
		c[i] = a

	#Smoothing
	d = array_smoother(np.transpose(c),smoothing)
	return d



def _core_figure_2(full_data,title,ylabel,yaxis=None,bound=None,lines=[],timing_window=[0,None],three_d=False):
	timing_step = list(range(full_data.shape[0]))
	if timing_window[1] is None:
		timing_window[1] = len(full_data)
	if yaxis is None:
		yaxis = range(len(full_data[0]))
	kwargs = {
		"z":full_data[:,timing_window[0]:timing_window[1]],
		"x":np.asarray(timing_step[timing_window[0]:timing_window[1]]),
		"y":np.asarray(yaxis),
	}
	if bound is not None:
		kwargs["zmin"]=bound[0]
		kwargs["zmax"]=bound[1]
	if three_d:
		data = [
			go.Surface(**kwargs)
		]
	else:
		data = [
		go.Heatmap(**kwargs)
		]
	layout = go.Layout(
		title=title,
		autosize=False,
		width=1000,
		height=500,
		margin=dict(
			l=65,
			r=50,
			b=65,
			t=90
		),
		xaxis=dict(
			title='Time',
			titlefont=dict(
				family='Courier New, monospace',
				size=18,
				color='#7f7f7f'
			)
		),
		yaxis=dict(
			title=ylabel,
			titlefont=dict(
				family='Courier New, monospace',
				size=18,
				color='#7f7f7f'
			)
		),
		shapes=list(map(lambda x : 
			{
			    'type': 'line',
			    'x0': timing_step[0],
			    'y0': x-0.5,
			    'x1': timing_step[-1],
			    'y1': x-0.5,
			    'line': {
				'color': 'rgb(55, 128, 55)',
				'width': 3,
			    },
			},lines)),
		)

	fig = go.Figure(data=data, layout=layout)
	return fig


