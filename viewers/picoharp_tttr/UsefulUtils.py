import win32clipboard
import numpy as np
from io import StringIO
import string
import IPython
import matplotlib
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import re


############################################
# 
# Miscellaneous functions
#
############################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

		
def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def change_mpl_dpi(dpi_set=108.8):
    rcParams = matplotlib.rcParams.copy()
    IPython.display.set_matplotlib_formats('png', facecolor='#FFFFFF', dpi=dpi_set)
    matplotlib.rcParams.update({'figure.figsize': rcParams['figure.figsize']})
	
def norm(a):
	return a/np.max(a)
	
def get_file_title(filepath):
	m = re.search('(\w+)[.].+$', filepath)
	return m.group(1)
		
		
#########################################
#
# Windows clipboard functions
#
#########################################
		
def paste_to_array():
    win32clipboard.OpenClipboard()
    rawData = win32clipboard.GetClipboardData()
    # Some Winspec snapins put in a bad character at the end...remove it if 
    # present.
    lastchar = rawData[-1]
    if (not is_number(lastchar)) and (lastchar not in string.whitespace):
        rawData = rawData[:-1]
    win32clipboard.CloseClipboard()
    
    #is the first line column titles?
    lines = rawData.split('\n')
    col =lines[0].split('\t')
      
    data = np.genfromtxt(StringIO(rawData),delimiter='\t')

    if sum(map(is_number, col)) < len(col):
        data = data[1:,:]
    else:
        col = []    
    
    return data, col

def copy_string(s):
	win32clipboard.OpenClipboard()
	win32clipboard.EmptyClipboard()
	win32clipboard.SetClipboardText(s)    
	win32clipboard.CloseClipboard()

def copy_array(a, rowdelim='\n', coldelim='\t'):
    dims = a.shape
    
    if len(dims) > 2:
        print('Unable to copy arrays with more than 2 dimensions')
        return 
        
    if len(dims) == 1:
        copy_string(rowdelim.join([str(x) for x in a]))
		
    if len(dims) == 2:
        s = ''
        for i in range(dims[0]):
            s = s + coldelim.join([str(x) for x in a[i]]) + rowdelim
        
        copy_string(s)
	
#########################################
#
# Smoothing functions for 1D data
#
#########################################
	
def fourier_smooth(a, num_points=5):
    a_fft = np.fft.rfft(a)
    a_fft[num_points:] = 0
    
    return np.fft.irfft(a_fft, n=a.shape[0])
   
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    #From:  http://wiki.scipy.org/Cookbook/SavitzkyGolay
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

	
################################################
#
# Matplotlib drawing and formatting functions
#
################################################
def draw_roi(x1, y1, x2, y2, **draw_params):
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    
    ax = plt.gca()

    # Form a path
    verts = [(x1, y1),
             (x1, y2),
             (x2, y2),
             (x2, y1),
             (0, 0)]
    path = Path(verts, codes)

    # Draw the BG region on the image
    patch = patches.PathPatch(path, **draw_params)
    ax.add_patch(patch)

def no_ticks():
	plt.xticks([])
	plt.yticks([])
	
def format_axis(axis_in, major=None, minor=None, direction='out', position=''):
	if axis_in == 'x':
		axis = plt.gca().xaxis
	elif axis_in == 'y':
		axis = plt.gca().yaxis
	
	
	if major != None:
		axis.set_major_locator(MultipleLocator(major))
	if minor != None:
		axis.set_minor_locator(MultipleLocator(minor))

	if len(direction) > 0:
		axis.set_tick_params(which='both', direction=direction)
	else:
		axis.set_tick_params(which='both', direction='out')

	if len(position) > 0:
		axis.set_ticks_position(position)
	else:
		if axis_in == 'x':
			axis.set_ticks_position('bottom')
		else:
			axis.set_ticks_position('left')

# This is a bit more of an advanced programming thing for my usage.  I basically 
# create objects for each data file and use them to keep track of the analysis 
# progression and steps.
class Bunch(object):
	def __init__(self, **kwds):
		self.__dict__.update(kwds)
	
def init_notebook(dpi=120, fontsize=10):
	# load plotting libraries.
	import matplotlib

	
	#matplotlib is the python library for making plots.  The below dictionary updates
	#change the default settings such as font size, tick size, etc.
	matplotlib.rcParams.update(
		{'font.sans-serif': 'Arial',
		 'font.size': fontsize,
		 'font.family': 'Arial',
		 'mathtext.default': 'regular',
		 'axes.linewidth': 0.3, 
		 'axes.labelsize': fontsize,
		 'xtick.labelsize': fontsize,
		 'ytick.labelsize': fontsize,     
		 'lines.linewidth': 0.5,
		 'legend.frameon': False,
		 'xtick.major.width': 0.3,
		 'xtick.minor.width': 0.3,
		 'ytick.major.width': 0.3,
		 'ytick.minor.width': 0.3,
		 'xtick.major.size': 3,
		 'ytick.major.size': 3,
		 'xtick.minor.size': 1,
		 'ytick.minor.size': 1
		})

	# This tells matplotlib to create the plots in an "inline" format that the 
	# web browser can display.
	#%matplotlib inline

	# This adjusts the DPI (pixel resolution of the plots displayed).  Increase the DPI 
	# for higher resolution images.
	import IPython
	IPython.display.set_matplotlib_formats('png', facecolor='#FFFFFF', dpi=dpi)
	
	# This sets the default size the plots generated by matplotlib.  It has to be called after the 
	# previous lines that sets the default DPI.
	# This needs to occur after the IPython call.
	matplotlib.rcParams.update(
		{'figure.figsize': (4,3)})
