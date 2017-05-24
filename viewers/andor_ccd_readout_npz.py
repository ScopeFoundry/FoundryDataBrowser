from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg

class AndorCCDReadoutNPZ(DataBrowserView):
    
    name = 'andor_ccd_readout_npz'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot(title="Andor CCD Spectrum")
        
        self.plotline = self.plot.plot()
        
    def is_file_supported(self, fname):
        return "andor_ccd_readout.npz" in fname


    def on_change_data_filename(self, fname):
        
        try:
            dat = self.dat = np.load(fname)
            self.spec = dat['spectrum'].sum(axis=0)
            self.plotline.setData(dat['wls'], self.spec)
        except Exception as err:
            self.plotline.setData(0)
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)