from ScopeFoundry.data_browser import DataBrowserView
import pyqtgraph as pg
import h5py
from qtpy import QtWidgets
import numpy as np

class PowerScanH5View(DataBrowserView):
    
    name = 'power_scan_h5'
    
    def is_file_supported(self, fname):
        return('power_scan' in fname) and ('.h5' in fname)
            
    def setup(self):
        
        self.settings.New('spec_index', dtype=int, initial=0)
        self.settings.spec_index.add_listener(self.on_spec_index_change)
        
        self.ui = QtWidgets.QGroupBox()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        
        self.ui.spec_index_doubleSpinBox = QtWidgets.QDoubleSpinBox()
        self.settings.spec_index.connect_bidir_to_widget(self.ui.spec_index_doubleSpinBox)
        self.ui.layout().addWidget(self.ui.spec_index_doubleSpinBox)
        
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.ui.layout().addWidget(self.graph_layout)
        
        self.power_plot = self.graph_layout.addPlot()
        self.power_plot.setLogMode(x=True, y=True)
        
        self.power_plotcurve = self.power_plot.plot([1],[1])
        
        self.power_plot_arrow = pg.ArrowItem()
        self.power_plot_arrow.setPos(0,0)
        self.power_plot.addItem(self.power_plot_arrow)
        
        self.graph_layout.nextRow()
        
        self.spec_plot = self.graph_layout.addPlot()
        self.spec_plotcurve = self.spec_plot.plot([0])
        
        
    def on_change_data_filename(self, fname=None):

        try:        
            self.dat = h5py.File(fname, 'r')
        
            if 'measurement/power_scan_df' in self.dat:
                self.H = self.dat['measurement/power_scan_df']
            else:
                self.H = self.dat['measurement/power_scan']
                
            H = self.H
            
            self.settings.spec_index.change_min_max(0, len(H['pm_powers'])-1)
            
            
            if 'integrated_spectra' in H:
                self.power_plot_y = np.array(H['integrated_spectra'])
                # to fix issues with log-log plotting, we shift negative data
                if np.any(self.power_plot_y < 0): 
                    self.power_plot_y -= np.min(self.power_plot_y) - 1
                self.power_plotcurve.setData(H['pm_powers'], self.power_plot_y)
            else:    
                self.power_plotcurve.setData(H['pm_powers'])

            self.settings['spec_index'] = 0
            self.on_spec_index_change()

        except Exception as err:
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
    
    def on_spec_index_change(self):
        ii = self.settings['spec_index']
        
        H = self.H
        if 'integrated_spectra' in H:
            print(H['pm_powers'][ii], H['integrated_spectra'][ii])
            self.power_plot_arrow.setPos(np.log10(H['pm_powers'][ii]), np.log10(self.power_plot_y[ii]))
            
        if 'spectra' in H:
            self.spec_plotcurve.setData(H['wls'], H['spectra'][ii])
        