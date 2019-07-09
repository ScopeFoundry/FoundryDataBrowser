'''
Created on May 20, 2019

@author: Edward Barnard, Benedikt Ursprung
'''


from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
from qtpy import QtWidgets, QtCore

import numpy as np
import h5py
from pyqtgraph import dockarea
from ScopeFoundry.widgets import RegionSlicer
from scipy.optimize import least_squares
from ScopeFoundry.logged_quantity import LQCollection


class PicoquantHistogramH5View(DataBrowserView):

    name = 'picoquant_histogram_h5'
          
    def is_file_supported(self, fname):
        if "picoharp_histogram.h5" in fname:
            self.m_base = 'measurement/{}/'.format('picoharp_histogram')
            self.h_base = 'hardware/{}/'.format('picoharp')
            return True
        elif "hydraharp_histogram.h5" in fname:
            self.m_base = 'measurement/{}/'.format('hydraharp_histogram')
            self.h_base = 'hardware/{}/'.format('hydraharp')
            return True
        else:
            return False
    
    def setup(self):
    
        self.ui = self.dockarea = dockarea.DockArea()
                
        # graph_dock
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.plotdock = self.dockarea.addDock(name='Histograms', position='right', 
                              widget=self.graph_layout) 
        self.plot = self.graph_layout.addPlot()
        self.plot.setLogMode(False, True)
        self.plotdata = self.plot.plot(pen='r')
        self.fit_line = self.plot.plot(pen='g')       
                
        #settings
        self.settings.New('fit_option',str,initial='tau_x_calc',
                                      choices = ('poly_fit','tau_x_calc','biexponential'))
        self.settings.New('chan', dtype=int, initial=0)
        self.settings.New('binning', dtype=int, initial=1, vmin=1)
        self.settings.New('time_unit', dtype=str, initial='ns')
        self.settings.New('norm_data', bool, initial = False)
        
        self.settings.fit_option.add_listener(self.fit_xy)        
        self.settings.chan.add_listener(self.update_display)
        self.settings.binning.add_listener(self.update_display)                
        self.settings.norm_data.add_listener(self.update_display)
            
        # data slicers
        self.x_slicer = RegionSlicer(self.plotdata,slicer_updated_func=self.update_display,
                                     name='x_slicer', initial=[10,20], activated=True)
        self.bg_slicer = RegionSlicer(self.plotdata,slicer_updated_func=self.update_display,
                                      name='bg_subtract', initial=[0,10], activated=False)
        
        
        
        #settings_dock        
        self.setdock = self.dockarea.addDock(name='Settings', position='left', 
                              widget=self.settings.New_UI()) 
        self.setdock.layout.addWidget(self.x_slicer.New_UI())
        self.setdock.layout.addWidget(self.bg_slicer.New_UI())
        
        
        # Metadata from file             
        self.posible_meta_data = ['ElapsedMeasTime','Tacq','Resolution','CountRate0','CountRate1',
                                  'Binning','SyncRate','SyncDivider','count_rate0','count_rate1',
                                  'elapsed_meas_time']
        self.meta_data_settings = LQCollection()
        for lqname in self.posible_meta_data:
            self.meta_data_settings.New(lqname, ro=True)
        self.meta_data_settings.New('sample', dtype=str, ro=True)    
        self.meta_data_ui = self.meta_data_settings.New_UI()
        self.setdock.layout.addWidget(QtWidgets.QLabel('<b>Meta data found</b>'))
        self.setdock.layout.addWidget(self.meta_data_ui)

        
    def update_display(self):
        x,y = self.get_xy(apply_use_x_slice=False)
        self.plotdata.setData(x,y)
        self.fit_xy()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            self.meas = H = self.dat[self.m_base]
            
            self.time_array = H['time_array'][:] * 1e-3 #ns
            self.histograms = H['time_histogram'][:].reshape(-1, len(self.time_array))           
                        
            n_chan = self.histograms.shape[0]
            self.settings.chan.change_min_max(0, n_chan-1)
            
            self.update_metadata()
            self.dat.close()
            
            self.update_display()
            
            
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    
    def update_metadata(self):
        '''if a possible meta  data setting is found in h5_file it will be displayed'''
        h5_hw_settings = self.dat[self.h_base+'/settings'].attrs
        h5_meas_settings = self.dat[self.m_base+'/settings'].attrs
        for i,lqname in enumerate(self.posible_meta_data):
            self.meta_data_ui.layout().itemAt(i,0).widget().show()
            self.meta_data_ui.layout().itemAt(i,1).widget().show()  
            if lqname in h5_hw_settings.keys():
                self.meta_data_settings[lqname] = h5_hw_settings[lqname]
            elif lqname in h5_meas_settings.keys():
                self.meta_data_settings[lqname] = h5_meas_settings[lqname]       
            else:
                self.meta_data_ui.layout().itemAt(i,0).widget().hide()
                self.meta_data_ui.layout().itemAt(i,1).widget().hide()    
                
        try:
            self.settings['sample'] = self.dat['app/settings'].attrs['sample']
        except KeyError:
            pass  


    def get_xy(self, apply_use_x_slice=True):
        '''
        returns data for fitting.
        '''
        y = 1.0*self.histograms[self.settings['chan']]
        x = self.time_array
        
        if self.bg_slicer.activated.val:
            bg = y[self.bg_slicer.s_].mean()
            y -= bg
            
        binning = self.settings['binning']
        if  binning> 1:
            x,y = bin_y_average_x(x, y, binning, -1, datapoints_lost_warning=False)   
            
        if apply_use_x_slice:
            x = x[self.x_slicer.s_]
            y = y[self.x_slicer.s_]
             
        if self.settings['norm_data']:
            y = norm(y)
                
        return (x,y)

    @QtCore.Slot()
    def fit_xy(self):
        x,y  = self.get_xy(apply_use_x_slice=True)
        print(x.shape, y.shape)
        fit_func_dict = {'poly_fit':  self.poly_fit_xy,
                         'tau_x_calc': self.tau_x_calc_xy,
                         'biexponential': self.fit_biexponential_xy}
        fit_option = self.settings['fit_option']
        self.xf,self.yf = fit_func_dict[fit_option](x,y)
    
    def tau_x_calc_xy(self,x,y):
        t = x.copy()
        t -= t.min()        
        tau = tau_x_calc(y, t)
        self.fit_line.setData(x,y)
        
        #gather result
        quantities = ['$\\tau_e$']
        numbers = '{0:1.1f}'.format(tau).split(" ")
        units = [self.settings['time_unit']]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True), title='tau_x_calc')

        return x,y

        
    def poly_fit_xy(self,x,y,deg=1):       
        coefs = poly_fit(x=x, y=y)
        t = x - x.min()
        fit = np.exp( np.poly1d(coefs)(t) )    
        self.fit_line.setData(x,fit)
        
        #gather result
        quantities = ['$A$','$\\tau$']
        numbers = '{0:1.1f} {1:1.1f}'.format(coefs[1],-1/coefs[0]).split(" ")
        units = ['-', self.settings['time_unit']]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True), title='poly_fit')
        
        return x,fit
    
    def fit_biexponential_xy(self,x,y):
        #bounds = self.biexponential_fit_bounds
        #bi_initial = self.biexponential_fit_initials
        bi_initial = [10,0.1,1,100]
        
        t = x - x.min()
                
        bi_res = least_squares(fun = biexponential_residuals,
                                 #bounds  = bounds,
                                 x0 = bi_initial, 
                                 args = (t, y))

        A0,tau0,A1,tau1 = bi_res.x
        A0,tau0,A1,tau1 = sort_biexponential_components(A0, tau0, A1, tau1)
        
        A0_norm,A1_norm = A0/(A0 + A1),A1/(A0 + A1)
        tau_m = A0_norm*tau0 + A1_norm*tau1
        fit = biexponential(bi_res.x, t)
        self.fit_line.setData(x,fit)
        #self.current_bi_exp_fit_res = bi_res.x


        quantities = ['$\\tau_0$','$\\tau_1$','$A_0$','$A_1$','$\\tau_m$']
        numbers = '{0:1.1f} {1:1.1f} {2:1.0f} {3:1.0f} {4:1.1f}'.format(tau0,tau1,A0_norm*100,A1_norm*100,tau_m).split(" ")
        time_unit = ''
        units = [time_unit, time_unit, '%', '%', time_unit]
        self.res_data_table = [[quantity, number, unit] for quantity, number, unit in zip(quantities,numbers,units)]
        self.x_slicer.set_label(_table2html(self.res_data_table, strip_latex=True),title='biexponential fit')
        
        return x,fit


def poly_fit(y,x,deg=1):
        mask = y > 0
        x = x[mask]
        y = y[mask]
        t = x.copy()
        t -= t.min()
        coefs = np.polyfit(t,np.log(y),deg)
        return coefs

def tau_x_calc(time_trace, time_array, x=0.6321205588300001):
    t = time_trace
    return time_array[np.argmin(np.abs(np.cumsum(t)/np.sum(t)-x))]        

def biexponential(params, t):
    '''
    params = [ A0, tau0, A1, tau1]    
    '''
    return params[0]*np.exp(-t/params[1]) + params[2]*np.exp(-t/params[3])
def biexponential_residuals(params, t, data):
    return biexponential(params,t) - data 
def fit_biexpontial(y, t,  bi_initial, bounds):
    bi_res = least_squares(fun = biexponential_residuals,
                                 bounds  = bounds,
                                 x0 = bi_initial, 
                                 args = (t, y))
    return bi_res.x    
def sort_biexponential_components(A0,tau0,A1,tau1):
    '''
    ensures that tau0 < tau1, also swaps values in A1 and A0 if necessary.
    '''
    A0 = np.atleast_1d(A0)
    tau0 = np.atleast_1d(tau0)
    A1 = np.atleast_1d(A1)
    tau1 = np.atleast_1d(tau1) 
    mask = tau0 < tau1
    mask_ = np.invert(mask)
    new_tau0 = tau0.copy()
    new_tau0[mask_] = tau1[mask_]
    tau1[mask_] = tau0[mask_]
    new_A0 = A0.copy()
    new_A0[mask_] = A1[mask_]
    A1[mask_] = A0[mask_]
    try:
        new_A0 = np.asscalar(new_A0)
        new_tau0 = np.asscalar(new_tau0)
        A1 = np.asscalar(A1)
        tau1 = np.asscalar(tau1)
    except ValueError:
        pass
    return new_A0,new_tau0,A1,tau1 #Note, generally A1,tau1 were also modified.


def norm(x):
    x_max = x.max()
    if x_max==0:
        return x*0.0
    else:
        return x*1.0/x_max
    
def bin_y_average_x(x, y, binning = 2, axis = -1, datapoints_lost_warning = True):
    '''
    y can be a n-dim array with length on axis `axis` equal to len(x)
    '''    
    new_len = int(x.__len__()/binning) * binning
    
    data_loss = x.__len__() - new_len
    if data_loss is not 0 and datapoints_lost_warning:
        print('bin_y_average_x() warining: lost final', data_loss, 'datapoints')
    
    def bin_1Darray(arr, binning=binning, new_len=new_len):
        return arr[:new_len].reshape((-1,binning)).sum(1)
    
    x_ = bin_1Darray(x) / binning
    y_ = np.apply_along_axis(bin_1Darray,axis,y)
    
    return x_, y_


def _table2html(data_table, strip_latex = True):
    text = '<table border="0">'
    for line in data_table:
        text += '<tr>'
        for element in line:
            text += '<td>{} </td>'.format(element)
        text += '</tr>'    
    text += '</table>'
    if strip_latex:
        text = text.replace('\\','').replace('$','').replace('_','')
    return text

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(PicoquantHistogramH5View(app))
    
    sys.exit(app.exec_())