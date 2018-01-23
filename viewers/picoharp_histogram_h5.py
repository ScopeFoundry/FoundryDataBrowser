from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np
import h5py

class PicoHarpHistogramH5View(DataBrowserView):

    name = 'picoharp_histogram_h5'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot()
        self.plotdata = self.plot.plot(pen='r')
        self.plot.setLogMode(False, True)
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            #rep_period_s = 1.0/self.dat['hardware/picoharp/settings/'].attrs['count_rate0']
            #print('rep period',rep_period_s)
            #time_bin_resolution = self.dat['hardware/picoharp/settings/'].attrs['Resolution']*1e-12
            
            #num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
            #print('num of hist chans', num_hist_chans)
            
            #print(self.dat['hardware/picoharp/settings/'].attrs['Resolution'])

            H = self.dat['measurement/picoharp_histogram/']
            time_array = np.array(H['time_array'])
            histogram= np.array(H['time_histogram'])
            
            self.plotdata.setData(1e-3*time_array,histogram)
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "picoharp_histogram.h5" in fname        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(PicoHarpHistogramH5View(app))
    
    sys.exit(app.exec_())