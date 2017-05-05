from ScopeFoundry.data_browser import DataBrowserView
import numpy as np
import h5py
import pyqtgraph as pg

class AugerSpectrumH5(DataBrowserView):

    name = 'auger_spectrum_h5'
    
    def setup(self):
        
        self.ui = self.graph_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graph_layout.addPlot(title="Auger Spectrum")
        
        self.display_chans = 7
        self.plot_setup()
        
    def on_change_data_filename(self, fname):
        
        try:
            self.dat = h5py.File(fname, 'r')
            
            self.H = self.dat['measurement/auger_spectrum']
            self.chan_data = self.H['chan_data']
            self.ke = self.H['ke']
            self.dwell_time = self.H['settings'].attrs['dwell']
            self.chan_Hz = np.zeros(self.chan_data.shape)
            self.chan_Hz = self.chan_data/self.dwell_time
            self.update_display()
            
        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
        
    def is_file_supported(self, fname):
        return "auger_spectrum.h5" in fname      
      
    def plot_setup(self):
        ''' create plots for channels and/or sum'''
        self.plot_lines = []
        for i in range(self.display_chans):
            color = pg.intColor(i)
            plot_line = self.plot.plot([0], pen=color)
            self.plot_lines.append(plot_line)
            #channel average
        plot_line = self.plot.plot([0], pen=color)
        self.plot_lines.append(plot_line)

    def update_display(self):    
        for i in range(self.display_chans):
            self.plot_lines[i].setData(self.ke[i,:],self.chan_Hz[i,:])
        #else:
        #    self.plot_lines[self.display_chans].setData(None)
