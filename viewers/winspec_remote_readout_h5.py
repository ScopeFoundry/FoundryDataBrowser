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
        self.infline = pg.InfiniteLine(movable=True, angle=90, label='x={value:0.2f}', 
           labelOpts={'position':0.8, 'color': (200,200,100), 'fill': (200,200,200,50), 'movable': True})         
        self.plot.addItem(self.infline)
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            try:
                self.H = self.dat['measurement/WinSpecRemoteReadout']
            except:
                self.H = self.dat['measurement/winspec_readout']
            self.wls = np.array(self.H['wls'])
            self.spectrum = self.H['spectrum']
            self.plotdata.setData(self.wls, np.squeeze(self.spectrum))
            self.infline.setValue([self.wls.mean(),0 ])
            
            self.center_wl = self.dat['hardware/acton_spectrometer/settings'].attrs['center_wl']
            print('acton center_wl', self.center_wl)
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return ("WinSpecRemoteReadout.h5" in fname) or  ("winspec_readout.h5" in fname)      
