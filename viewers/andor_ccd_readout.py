from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg

class AndorCCDReadout(DataBrowserView):
    
    name = 'andor_ccd_readout'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot(title="Andor CCD Spectrum")
        
        self.plotline = self.plot.plot()
        
    def is_file_supported(self, fname):
        return "andor_ccd_readout.npz" in fname or "andor_ccd_readout.h5" in fname


    def on_change_data_filename(self, fname):
        if fname is None:
            fname = self.databrowser.settings['data_filename']

        try:
            if '.npz' in fname: 
                dat = self.dat = np.load(fname)
                self.spec = dat['spectrum'].sum(axis=0)
                self.wls = np.array(self.dat['wls'])
            elif '.h5' in fname:
                dat = self.dat = h5py.File(fname)
                self.M = dat['measurement/andor_ccd_readout']
                self.spec = np.array(self.M['spectrum']).sum(axis=0)
                self.wls = np.array(self.M['wls'])
                
            self.plotline.setData(self.wls, self.spec)
        except Exception as err:
            self.plotline.setData(0)
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)