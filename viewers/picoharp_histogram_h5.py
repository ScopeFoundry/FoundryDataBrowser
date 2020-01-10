from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np
import h5py
from pyqtgraph import dockarea

class PicoHarpHistogramH5View(DataBrowserView):

    name = 'picoharp_histogram_h5'
    
    def setup(self):
        
        self.settings.New('rebin', dtype=bool, initial=False)
        self.settings.New('rebin_time', dtype=int, initial=4, unit='ps', choices=tuple(4*2**np.arange(0,10)))
        
        
        # settings from file
        self.settings.New('sample', dtype=str, ro=True)
        self.settings.New('elapsed_meas_time', dtype=float, unit='ms', ro=True)
        self.settings.New('Tacq', dtype=float, unit='s', ro=True)
        self.settings.New('Resolution', dtype=float, unit='ps', ro=True)
        self.settings.New('count_rate0', dtype=float, unit='Hz', ro=True)    
        self.settings.New('count_rate1', dtype=float, unit='Hz', ro=True)
        
        self.ui = self.dockarea = dockarea.DockArea()
        
        self.setdock = self.dockarea.addDock(name='Settings', position='left', 
                              widget=self.settings.New_UI()) 

        self.graph_layout = pg.GraphicsLayoutWidget()
        self.plotdock = self.dockarea.addDock(name='Picoharp Histogram', position='right', 
                              widget=self.graph_layout) 

        
        self.plot = self.graph_layout.addPlot()
        self.plotdata = self.plot.plot(pen='r')
        self.plot.setLogMode(False, True)
        
        self.settings.rebin.add_listener(self.on_rebin)
        self.settings.rebin_time.add_listener(self.on_rebin)

        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            #rep_period_s = 1.0/self.dat['hardware/picoharp/settings/'].attrs['count_rate0']
            #print('rep period',rep_period_s)
            #time_bin_resolution = self.dat['hardware/picoharp/settings/'].attrs['Resolution']*1e-12
            
            #num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
            #print('num of hist chans', num_hist_chans)
            
            #print(self.dat['hardware/picoharp/settings/'].attrs['Resolution'])

            self.meas = H = self.dat['measurement/picoharp_histogram/']
            self.time_array_ps = np.array(H['time_array'])
            self.histogram = np.array(H['time_histogram'])
            
            self.update_metadata()
            self.on_rebin()
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    def update_metadata(self):
        self.settings['sample'] = self.dat['app/settings'].attrs['sample']
        self.settings['elapsed_meas_time'] = float(np.array(self.dat['measurement/picoharp_histogram/elapsed_meas_time']))
        self.settings['Tacq'] = self.dat['hardware/picoharp/settings'].attrs['Tacq']
        self.settings['Resolution'] = self.dat['hardware/picoharp/settings'].attrs['Resolution']
        self.settings['count_rate0'] = self.dat['hardware/picoharp/settings'].attrs['count_rate0']
        self.settings['count_rate1'] = self.dat['hardware/picoharp/settings'].attrs['count_rate1']
        
    def on_rebin(self):
        S = self.settings
        if S['rebin']:
            dt = self.time_array_ps[1] - self.time_array_ps[0] 
            
            bin_num = S['rebin_time']/dt            
            
            if bin_num > 1: # combine bins in time
                N = int(np.floor(len(self.time_array_ps)/bin_num)*bin_num)
                
                rebin_time_array_ps = self.time_array_ps[:N:bin_num]
                rebin_histogram = self.histogram[:N].reshape(-1, bin_num).sum(axis=1)
            else: # Interpolate time bins by filling with normalized nearest neighbor
                m = int(1/bin_num)
                T = self.time_array_ps
                rebin_time_array_ps = np.interp(np.arange(len(T)*m)*1.0/m, np.arange(len(T)),T)
                rebin_histogram = np.zeros_like(rebin_time_array_ps)
                
                for i in range(m):
                    rebin_histogram[i::m] = self.histogram*1.0/m
                
            self.plotdata.setData(1e-3*rebin_time_array_ps, rebin_histogram)                
        else:
            self.plotdata.setData(1e-3*self.time_array_ps,self.histogram)
            
    def is_file_supported(self, fname):
        return "picoharp_histogram.h5" in fname        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(PicoHarpHistogramH5View(app))
    
    sys.exit(app.exec_())