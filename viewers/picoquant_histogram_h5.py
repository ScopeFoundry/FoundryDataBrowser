'''
Created on May 20, 2019

@author: Edward Barnard, Benedikt Ursprung
'''

from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
from qtpy import QtWidgets, QtCore, QtGui

import numpy as np
import h5py
from ScopeFoundry.widgets import RegionSlicer

from FoundryDataBrowser.viewers.plot_n_fit import PlotNFit, MonoExponentialFitter, BiExponentialFitter, TauXFitter, SemiLogYPolyFitter
    

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
        ## ui and graph plot
        self.plot_n_fit = PlotNFit(
                             fitters=[
                                      SemiLogYPolyFitter(),
                                      MonoExponentialFitter(),
                                      BiExponentialFitter(),
                                      TauXFitter(),
                                      ],
                             )  
        self.ui = self.dockarea = self.plot_n_fit.get_docks_as_dockarea()
        self.plot_n_fit.plot.setLogMode(False, True)  
        
            
        # data slicers
        plot_data = self.plot_n_fit.data_lines[0]
        self.x_slicer = RegionSlicer(plot_data, 
                                     brush = QtGui.QColor(0,255,0,50), 
                                     name='x_slicer', initial=[10,20], activated=True)
        self.x_slicer.region_changed_signal.connect(self.update_display)

        self.bg_slicer = RegionSlicer(plot_data,
                                      brush = QtGui.QColor(255,255,255,50), 
                                      name='bg_subtract', initial=[0,10], activated=False)     
        self.bg_slicer.region_changed_signal.connect(self.update_display)
        
        self.plot_n_fit.settings_layout.insertWidget(0,self.x_slicer.New_UI())
        self.plot_n_fit.settings_layout.insertWidget(1,self.bg_slicer.New_UI())

        
                        
        ##settings dock
        self.settings.New('chan', dtype=int, initial=0)
        self.settings.New('binning', dtype=int, initial=1, vmin=1)
        self.settings.New('time_unit', dtype=str, initial='ns')
        self.settings.New('norm_data', bool, initial = False)
        self.settings.New('roll_data', int, initial = 0)
        for lqname in ['chan', 'binning', 'roll_data', 'norm_data']:
            getattr(self.settings, lqname).add_listener(self.update_display)
        
        
        self.setdock = self.dockarea.addDock(name='Data Settings', position='below', 
                                             relativeTo=self.plot_n_fit.settings_dock,
                                             widget=self.settings.New_UI()) 
        
     
        # Metadata from file
        self.posible_meta_data = ['ElapsedMeasTime','Tacq','Resolution','CountRate0','CountRate1',
                                  'Binning','SyncRate','SyncDivider','count_rate0','count_rate1',
                                  'elapsed_meas_time', 'sample']
                
        self.setdock.layout.addWidget(QtWidgets.QLabel('<h3>Meta data </h3>'))
        self.meta_data_label = QtWidgets.QLabel()
        self.setdock.layout.addWidget(self.meta_data_label)


        # Just for appearance
        VSpacerItem = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, 
                                        QtWidgets.QSizePolicy.Expanding)
        self.setdock.layout.addItem(VSpacerItem)
        self.setdock.setStretch(1, 1)
        self.plot_n_fit.settings_dock.setStretch(1, 1)
        self.plot_n_fit.settings_dock.raiseDock()

    @QtCore.Slot()
    def update_display(self):
        x,y = self.get_xy(apply_use_x_slice=False)
        self.plot_n_fit.update_data(x, y, n_plot=0, is_fit_data=False)
        
        x_fit_data, y_fit_data = self.get_xy(apply_use_x_slice=True)
        self.plot_n_fit.update_fit_data(x_fit_data, y_fit_data)
        
        text = self.plot_n_fit.result_message
        title = self.plot_n_fit.fit_options.val
        self.x_slicer.set_label(text, title)
        
        
    def on_change_data_filename(self, fname):
        try:
            self.dat = h5py.File(fname, 'r')
            self.meas = H = self.dat[self.m_base]
            
            self.time_array = H['time_array'][:] * 1e-3 #ns
            self.histograms = H['time_histogram'][:].reshape(-1, len(self.time_array)) #force shape (Nchan, Nbins)           
                        
            n_chan = self.histograms.shape[0]
            self.settings.chan.change_min_max(0, n_chan-1)
            
            self.update_metadata()
            self.dat.close()
            
            self.update_display()
            
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    
    def update_metadata(self):
                
        app_set = self.dat['app/settings']
        hw_set = self.dat[self.h_base+'/settings']
        meas_set = self.dat[self.m_base+'/settings']

        data_table = []        
        for settings in [app_set, hw_set, meas_set]:
            for lqname in self.posible_meta_data:
                if lqname in settings.attrs.keys():
                    val = settings.attrs[lqname]
                    unit = ''
                    if lqname in settings['units'].attrs.keys():
                        unit = settings['units'].attrs[lqname]
                    data_table.append([lqname, val, unit])

        html_table = _table2html(data_table, False)
        self.meta_data_label.setText(html_table)
        

    def get_xy(self, apply_use_x_slice=True):
        '''
        returns data as configurate.
        '''
        try:
            y = 1.0*self.histograms[self.settings['chan']]
            x = self.time_array
            
            R = self.settings['roll_data']
            if R!=0: y = np.roll(y,R,-1)
            
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
        except:
            x = np.arange(120)/12
            y = np.exp(-x / 10.0) + 0.001 * np.random.rand(len(x))
        
        return (x,y)
        

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


def _table2html(data_table, strip_latex = True, header=[]):
    text = '<table border="0" alignment="center" >'
    if len(header) == len(data_table[0]):
        text += '<tr>'
        for element in header:
            text += '<th>{} </th>'.format(element)
        text += '</tr>'  
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