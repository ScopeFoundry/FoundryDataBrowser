from ScopeFoundry.data_browser import DataBrowserView
from FoundryDataBrowser.viewers.plot_n_fit import PlotNFit, PeakUtilsFitter
import numpy as np
import h5py

class AndorCCDReadout(DataBrowserView):
    
    name = 'andor_ccd_readout'
    
    def setup(self):
        
        self.plot_n_fit = PlotNFit(fitters=[PeakUtilsFitter()])                
        self.ui = self.plot_n_fit.get_docks_as_dockarea()
        self.plot_n_fit.settings['fit_options'] = 'DisableFit'

        
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
                
            self.plot_n_fit.update_data(self.wls, self.spec)
            
        except Exception as err:
            self.plot_n_fit.update_data([0,1,2,3],[1,3,2,4])
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)