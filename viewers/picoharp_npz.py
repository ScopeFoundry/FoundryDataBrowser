from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
import pyqtgraph as pg
import numpy as np

class PicoHarpNPZView(DataBrowserView):

    name = 'picoharp_npz'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot()
        self.plotdata = self.plot.plot(pen='r')
        self.plot.setLogMode(False, True)
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = np.load(fname)

            rep_period_s = 1.0/self.dat['picoharp_count_rate0']
            #print('rep period',rep_period_s)
            time_bin_resolution = self.dat['picoharp_Resolution']*1e-12
            #print('resolution (s)', time_bin_resolution)
            #print('num of hist chans', rep_period_s/time_bin_resolution)
            num_hist_chans = int(np.ceil(rep_period_s/time_bin_resolution))
            #print('num of hist chans', num_hist_chans)
            
            
            self.plotdata.setData(1e-3*self.dat['time_array'][0:num_hist_chans],
                                  self.dat['time_histogram'][0:num_hist_chans])
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "_picoharp.npz" in fname        

if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(PicoHarpNPZView(app))
    
    sys.exit(app.exec_())