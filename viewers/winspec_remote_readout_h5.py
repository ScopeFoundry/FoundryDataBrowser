from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg

class WinSpecRemoteReadoutView(DataBrowserView):

    name = 'winspec_remote_readout_h5'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot()
        self.plotdata = self.plot.plot()
        self.infline = pg.InfiniteLine(movable=True, angle=90, label='x={value:0.2f}', 
           labelOpts={'position':0.8, 'color': (200,200,100), 'fill': (200,200,200,50), 'movable': True})         
        self.plot.addItem(self.infline)
        
    def on_change_data_filename(self, fname):       
        try:
            self.load_data(fname)

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            print("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
        finally:        
            self.update_display()
            
    def load_data(self, fname):
        if hasattr(self, 'dat'):
            del self.dat
            del self.wls
            del self.spectrum

        self.dat = h5py.File(fname, 'r')

        self.H = self.dat['measurement/winspec_readout']
        self.wls = np.array(self.H['wls'])
        self.spectrum = self.H['spectrum']

        self.center_wl = self.dat['hardware/acton_spectrometer/settings'].attrs['center_wl']
        print('acton center_wl', self.center_wl)
        
                
    def update_display(self):
        self.plotdata.setData(self.wls, np.squeeze(self.spectrum))
        self.infline.setValue([self.wls.mean(),0 ])        
                    
    def is_file_supported(self, fname):
        return ("WinSpecRemoteReadout.h5" in fname) or  ("winspec_readout.h5" in fname)      
