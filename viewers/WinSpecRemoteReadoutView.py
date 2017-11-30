from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg

class WinSpecRemoteReadoutView(DataBrowserView):

    name = 'WinSpecRemoteReadout'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot()
        self.plotdata = self.plot.plot()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            try:
                self.H = self.dat['measurement/WinSpecRemoteReadout']
            except:
                self.H = self.dat['measurement/winspec_readout']
            self.wls = self.H['wls']
            self.spectrum = self.H['spectrum']
            self.plotdata.setData(self.wls, np.squeeze(self.spectrum))
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return ("WinSpecRemoteReadout.h5" in fname) or  ("winspec_readout.h5" in fname)      
